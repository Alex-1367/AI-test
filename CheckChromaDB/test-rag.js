import { OllamaEmbedding } from './ollama-embeddings.js';
import { createInterface } from 'readline';

async function main() {
  const rag = new OllamaEmbedding();
  
  console.log('🚀 Starting Interactive RAG...\n');
  await rag.initialize();
  
  const count = await rag.countDocuments();
  console.log(`📊 Collection has ${count} documents`);

  const readline = createInterface({
    input: process.stdin,
    output: process.stdout
  });

  console.log('\n' + '='.repeat(50));
  console.log('🤖 RAG System Ready! Ask me anything.');
  console.log('='.repeat(50));

  const askQuestion = () => {
    readline.question('\n❓ Your question: ', async (query) => {
      if (query.toLowerCase() === 'quit') {
        readline.close();
        return;
      }

      const result = await rag.query(query);
      if (result) {
        console.log('\n💡 Answer:', result.answer);
      }

      askQuestion();
    });
  };

  askQuestion();
}

main().catch(console.error);