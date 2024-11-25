import sys

sys.path += ['../']
import torch
import os
from utils.util import (
    barrier_array_merge,
    StreamingDataset,
    EmbeddingCache,
)
from data.tokenizing import GetProcessingFn
from model.models import MSMarcoConfigDict
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj
import re

from transformers import RobertaTokenizer #Added this!
import pickle


torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

#------------------Load mappings--------------------------------------------------------
pid2offset_path = "/home/scur2878/ConvDR/datasets/cast-shared/tokenized_v2/pid2offset.pickle"
offset2pid_path = "/home/scur2878/ConvDR/datasets/cast-shared/tokenized_v2/offset2pid.pickle"

with open(pid2offset_path, "rb") as f:
    pid2offset = pickle.load(f)

with open(offset2pid_path, "rb") as f:
    offset2pid = pickle.load(f)

# testing this out -> OBV wrong since it is for TREC-CAR dataset
# with open("datasets/cast-shared/car_idx_to_id.pickle", "rb") as f:
#     car_idx_to_id = pickle.load(f)
#--------------------------------------------------------------------------


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path

    config, tokenizer, model = None, None, None
    if args.model_type != "dpr":
        config = configObj.config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="MSMarco",
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        #Using RobertaTokenizer directly!
        tokenizer = RobertaTokenizer(                                       
            vocab_file=os.path.join(checkpoint_path, "vocab.json"),
            merges_file=os.path.join(checkpoint_path, "merges.txt"),
            do_lower_case=True,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = configObj.model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:  # dpr
        model = configObj.model_class(args)
        saved_state = load_states_from_checkpoint(checkpoint_path)
        model_to_load = get_model_obj(model)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)

    model.to(args.device)
    logger.info("Inference parameters %s", args)

    # -------------------------(Verify Model loading)---------------------------------
    if model is None:
        logger.error("Model loading failed. Check the checkpoint path.")
    else:
        logger.info("Model loaded successfully from checkpoint.")
    # ---------------------------------------------------------------
    

    if args.local_rank != -1 and torch.cuda.device_count() > 1:  # Added this (and torch.cuda.device_count() > 1): Only if running distributed training & Multiple GPUs
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(
    args,
    model,
    train_dataloader,
    is_query_inference=True,
):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size


    # # Check if train_dataloader has data by attempting to get the first batch
    # try:
    #     first_batch = next(iter(train_dataloader))
    #     logger.info("train_dataloader is loaded and contains data.")
    # except StopIteration:
    #     logger.error("train_dataloader is empty. Check data sources or preprocessing.")
    #     return None, None  # Return appropriate values or handle the empty case

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()
    logger.info("Model set to evaluation mode.") #Added this!

    for batch in tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.local_rank not in [-1, 0],
                      position=0,
                      leave=True):

        # if args.local_rank in [-1, 0]:  # Added this!
        #     logger.debug(f"Processing batch ...")
        
        # Inspecting  the structure of the batch------------------------------------
        # print(f"Batch length: {len(batch)}")
        # for i, element in enumerate(batch):
        #     print(f"Element {i}:")
        #     if isinstance(element, torch.Tensor):
        #         print(f"  Type: Tensor")
        #         print(f"  Shape: {element.shape}")
        #         print(f"  Data (first 5 elements): {element[:5].cpu().numpy()}")  # Preview the first 5 values
        #     else:
        #         print(f"  Type: {type(element)}")
        #         print(f"  Content: {element}")
        
        #Get the following example output :--------------------------------------------
        # Batch length: 4
        # Element 0: contains the tokenized input sequences for the batch. 
        # Type: Tensor
        # Shape: torch.Size([64, 512])
        # Data (first 5 elements): [[    0  4148   971 ...     0     0     0]
        # [    0  4993     5 ...     0     0     0]
        # [    0   250 32638 ...     0     0     0]
        # [    0  1121   644 ...     0     0     0]
        # [    0   250   889 ...     0     0     0]]
        # Element 1:the attention_mask. It indicates which tokens are valid (True) and which are padding (False). Padding tokens are not attended to during processing.
        # Type: Tensor
        # Shape: torch.Size([64, 512])
        # Data (first 5 elements): [[ True  True  True ... False False False]
        # [ True  True  True ... False False False]
        # [ True  True  True ... False False False]
        # [ True  True  True ... False False False]
        # [ True  True  True ... False False False]]
        # Element 2: distinguishes between different parts of input sequences
        # Type: Tensor
        # Shape: torch.Size([64, 512])
        # Data (first 5 elements): [[1 1 1 ... 0 0 0]
        # [1 1 1 ... 0 0 0]
        # [1 1 1 ... 0 0 0]
        # [1 1 1 ... 0 0 0]
        # [1 1 1 ... 0 0 0]]
        # Element 3: the IDs tensor, which uniquely identifies the passages or queries in the batch.
        # Type: Tensor
        # Shape: torch.Size([64])
        # Data (first 5 elements): [0 1 2 3 4]
        #---------------------------------------------------

        #that's why it indexes element 3 from the batch,explanation above
        # idxs = batch[3].detach().numpy()  # [#B] #the offsets provided by dataloader e.g. [0,1,2,3,4]
        idxs = [offset2pid[idx] for idx in batch[3].detach().numpy()]  # Map sequential IDs to original passage IDs

        # Map the offsets to the real document IDs using offset2pid
        # mapped_ids = [offset2pid[offset] for offset in idxs] #REAL doc IDs by mapping the offsets
        # logger.info(f"Remapped Document IDs (using offset2pid): {mapped_ids[:5]}")

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()
            }
            if is_query_inference:
                # embs = model.module.query_emb(**inputs)
                embs = model.query_emb(**inputs) if not hasattr(model, 'module') else model.module.query_emb(**inputs) #Added these!
            else:
                # embs = model.module.body_emb(**inputs)
                embs = model.body_emb(**inputs) if not hasattr(model, 'module') else model.module.body_emb(**inputs) #Added these!

        # if embs is None:
        #     logger.error("Embeddings not generated. Check model output.")
        # else:
        #     logger.debug("Embeddings generated successfully for the batch.") #Added this!

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                # embedding2id.append(mapped_ids)  # Use mapped IDs instead of idxs
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            # embedding2id.append(mapped_ids)  # Use mapped IDs instead of idxs
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)

    logger.info(f"Generated embeddings shape: {embedding.shape}")
    logger.info(f"Generated embedding IDs shape: {embedding2id.shape}")
    logger.info(f"Sample embedding ID: {embedding2id[:5]}")
    logger.info(f"Sample embedding: {embedding[:5]}")

    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args,
                       model,
                       fn,
                       prefix,
                       f,
                       is_query_inference=True,
                       merge=True):
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier()  # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        inference_dataloader,
        is_query_inference=is_query_inference,
        )

    # Confirm embedding generation
    logger.info(f"Embedding shape: {_embedding.shape if _embedding is not None else 'None'}")
    logger.info(f"Embedding2ID shape: {_embedding2id.shape if _embedding2id is not None else 'None'}")

    logger.info(f"Embedding IDs: {_embedding2id[:5]}")
    logger.info(f"Embeddings: {_embedding[:5]}")

    if _embedding is None or _embedding2id is None:
        logger.error("No embeddings generated. Exiting without saving.")
        return None, None
    
    logger.info("merging embeddings")

    # preserve to memory
    full_embedding = barrier_array_merge(args,
                                         _embedding,
                                         prefix="msmarco_" + prefix + "_emb_p_",
                                         load_cache=False,
                                         only_load_in_master=True,
                                         merge=merge)
    full_embedding2id = barrier_array_merge(args,
                                            _embedding2id,
                                            prefix="msmarco_" + prefix + "_embid_p_",
                                            load_cache=False,
                                            only_load_in_master=True,
                                            merge=merge)
    
    # Check if files exist after merging and saving
    output_files = os.listdir(args.output_dir)
    if output_files:
        logger.info(f"Embeddings saved successfully in directory: {args.output_dir}")
    else:
        logger.error("Embeddings were not saved. Check save function or permissions.")

    return full_embedding, full_embedding2id


def generate_new_ann(
    args,
    checkpoint_path,
):

    _, __, model = load_model(args, checkpoint_path)
    merge = False

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.data_dir,
                                           "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=False),
            "passage_",
            emb,
            is_query_inference=False,
            merge=merge)
    logger.info("***** Done passage inference *****")


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the tokenized passage ids.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        required=True,
        help="Checkpoint of the ad hoc retriever",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MSMarcoConfigDict.keys()),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where cached data will be written",
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=64,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--server_ip",
        type=str,
        default="",
        help="For distant debugging.",
    )

    parser.add_argument(
        "--server_port",
        type=str,
        default="",
        help="For distant debugging.",
    )

    args = parser.parse_args()

    return args


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Setup logging with DEBUG level for main process
    logging_level = logging.DEBUG if args.local_rank in [-1, 0] else logging.WARN
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_level,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )


def ann_data_gen(args):

    logger.info("start generate ann data")
    generate_new_ann(
        args,
        args.checkpoint,
    )

    if args.local_rank != -1:
        dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()
