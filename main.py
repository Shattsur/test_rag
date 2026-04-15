import argparse
from rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG система для вопросов по документам")
    parser.add_argument("--query", type=str, help="Вопрос к системе")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    args = parser.parse_args()
    
    rag = RAGPipeline()
    
    if args.interactive:
        print("RAG система запущена. Введите 'exit' для выхода.")
        while True:
            q = input("\nВаш вопрос: ")
            if q.lower() in ["exit", "quit"]:
                break
            answer, sources = rag.generate(q, return_sources=True)
            print(f"\nОтвет: {answer}")
            print("Источники:")
            for s in sources:
                print(f"  - {s['source']}")
    elif args.query:
        answer, sources = rag.generate(args.query, return_sources=True)
        print(f"\nОтвет: {answer}\n")
        print("Источники:")
        for s in sources:
            print(f"  - {s['source']}: {s['text'][:100]}...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()