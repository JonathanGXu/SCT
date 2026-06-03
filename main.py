import argparse
import os
import json
import tqdm
import asyncio

from Modules.GraphRAG_EM import Graphrag_em
from Data.Datacleaner import DataLoader
from Modules.LLM import qwen_chat_completion
from Modules.LightRAG import LightRAGQA,HotpotQADataset
from Data.evaluate import generate_prediction_file,SimplifiedEvaluator
from Data.llm_baseline import run_llm_baseline
from config import DeepSeek_model

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphRAG arguments.')

    parser.add_argument("--datadir", type=str,
                        help= "e.g., Data/Reproduce")
    parser.add_argument("--dataset",type=str,
                        help= "e.g., [ 2wikimultihopqa, hotpotqa, musique, sample ]")

    parser.add_argument("--model",type=str,
                        help="The model used in test. [ LLM , GraphRAG-EM]")

    parser.add_argument("--llm_mode", type=str, default="zero", choices=["zero", "gold", "retrieval"])

    parser.add_argument('--agent_llm_model',dest='agent_llm_model',type=str,
                        help='agent llm model for build graph. e.g., qwen ')

    parser.add_argument('--agent_num',dest='agent_num',type=int)

    parser.add_argument('--embedding_model', dest='embedding_model', type=str,
                        help=' e.g., text-embedding-v4 ')

    parser.add_argument('--qa_llm_model', dest='qa_llm_model', type=str,
                        help='e.g., qwen ')

    parser.add_argument("--pred_dir",dest='pred_dir',type=str)

    parser.add_argument("--use_ollama", dest='use_ollama', action='store_true')

    parser.add_argument("--ollama_chat_model", dest='ollama_chat_model', type=str)

    parser.add_argument("--ollama_embed_model", dest='ollama_embed_model', type=str)

    parser.add_argument("--use_deepseek", dest='use_deepseek', action='store_true')

    parser.set_defaults(
                        datadir='Data/Reproduce',
                        dataset="sample",
                        model = "GraphRAG-EM",
                        llm_mode = "zero",
                        agent_llm_model="qwen-flash-2025-07-28",
                        agent_num= 3,
                        embedding_model = "ollama-embed",
                        qa_llm_model = "qwen-flash-2025-07-28",
                        pred_dir = 'Prediction',
                        use_ollama = False,
                        ollama_chat_model = "glm-5.1:cloud",
                        ollama_embed_model = "mxbai-embed-large",
                        use_deepseek = True,
                        )

    return parser.parse_args()

def load_data(datadir, dataset):
    print("Loading data...")
    data = DataLoader(datadir, dataset)
    qa_data = data.qa_data
    corpus = data.corpus

    processed_data = data.preprocess_structured_text(corpus)

    full_clean_text = [text['original_text'] for text in processed_data]

    return full_clean_text,qa_data


def run_graphrag_em():
    print("Beginning GraphRAG-GEM...")
    args = arg_parse()

    if args.use_deepseek:
        if args.agent_llm_model.startswith("qwen"):
            args.agent_llm_model = DeepSeek_model
        if args.qa_llm_model.startswith("qwen"):
            args.qa_llm_model = DeepSeek_model

    if args.use_deepseek:
        model_tag = f"deepseek-{args.agent_llm_model}"
    elif args.use_ollama:
        if ":cloud" in args.ollama_chat_model:
            model_tag = f"ollama-{args.ollama_chat_model[:-6]}"
        else:
            model_tag = f"ollama-{args.ollama_chat_model}"
    else:
        model_tag = args.agent_llm_model

    if args.model == "LLM":
        pred_file_path = os.path.join(args.pred_dir, f"{args.dataset}_{args.model}-{args.llm_mode}_{model_tag}.json")
    else:
        pred_file_path = os.path.join(args.pred_dir, f"{args.dataset}_{args.model}_{model_tag}.json")
    text_corpus, qa = load_data(args.datadir, args.dataset)

    if args.model == "GraphRAG-EM":

        graphrag = Graphrag_em(
            args.agent_num, args.agent_llm_model, args.embedding_model, args.qa_llm_model,
            use_ollama=args.use_ollama,
            ollama_chat_model=args.ollama_chat_model,
            ollama_embed_model=args.ollama_embed_model,
            use_deepseek=args.use_deepseek
        )

        graphrag.build_graph(text_corpus)

        predictions = {}
        for item in tqdm.tqdm(qa, desc="results"):
            question_id = item["id"]
            question = item["question"]


            answer, evidence = graphrag.qa(question)

            predictions[question_id] = {
                "answer": answer
            }

            print(answer)

        with open(pred_file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)


    elif args.model == "LLM":
        print(f"🚀 Running LLM baseline [{args.llm_mode}] ...")


        data_loader = DataLoader(args.datadir, args.dataset)
        raw_corpus = data_loader.corpus

        predictions = run_llm_baseline(
            qa_data=qa,
            corpus=raw_corpus,
            mode=args.llm_mode,
            dataset_name=args.dataset,
            qa_model=args.qa_llm_model,
            use_ollama=args.use_ollama,
            ollama_chat_model=args.ollama_chat_model,
            use_deepseek=args.use_deepseek,
            retrieval_top_k=5
        )

        with open(pred_file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)


    elif args.model == "LightRAG":

        rag_system = LightRAGQA(use_local_model=True)


        asyncio.run(rag_system.add_documents(text_corpus))

        predictions = {}
        for item in tqdm.tqdm(qa, desc="results"):
            question_id = item["id"]
            question = item["question"]
            supporting_facts = item["supporting_facts"]
            evidences = item["evidences"]

            answer = asyncio.run(rag_system.answer_question(question))
            evidence_candidates = rag_system.retrieve_evidence_candidates(question)


            predictions[question_id] = {
                "answer": answer,
                "evidence_candidates": evidence_candidates
            }

        with open(pred_file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    run_graphrag_em()
