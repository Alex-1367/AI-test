import { ChromaClient } from './chroma-client.js';

export class OllamaEmbedding {
  constructor(baseURL = 'http://localhost:11434', model = 'qwen2.5-coder:1.5b') {  // Using your model for embeddings too
    this.baseURL = baseURL;
    this.model = model;
  }

  async getEmbedding(text) {
    const response = await fetch(`${this.baseURL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,  // Your qwen2.5-coder model
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.status}`);
    }

    const data = await response.json();
    return data.embedding;
  }

  async getEmbeddings(texts) {
    return Promise.all(texts.map(text => this.getEmbedding(text)));
  }
}

class RAGApplication {
  constructor() {
    this.chroma = new ChromaClient('http://localhost:8000');
    this.embedding = new OllamaEmbedding('http://localhost:11434', 'qwen2.5-coder:1.5b');
    this.llmModel = 'qwen2.5-coder:1.5b';  // Your model for generation
    this.collectionName = 'knowledge_base';
    this.collectionId = null;
  }

  async initialize() {
    const collection = await this.chroma.getOrCreateCollection(this.collectionName, {
      space: 'cosine',
      description: 'Knowledge base for RAG application'
    });
    this.collectionId = collection.id;
    console.log(`✅ Collection ready (ID: ${this.collectionId})`);
  }

  async addDocuments(docs) {
    console.log(`📝 Generating embeddings for ${docs.length} documents...`);
    
    // Use qwen2.5-coder for embeddings
    const embeddings = await this.embedding.getEmbeddings(docs.map(d => d.text));

    console.log(`📊 Adding to ChromaDB...`);
    
    await this.chroma.addDocuments(
      this.collectionId,
      docs.map(d => d.text),
      docs.map((_, i) => `doc_${Date.now()}_${i}`),
      embeddings,
      docs.map(d => d.metadata || {})
    );
    
    console.log(`✅ Added ${docs.length} documents`);
  }

  async query(question) {
    console.log(`🔍 Generating embedding for question...`);
    
    // Use SAME model for question embedding
    const questionEmbedding = await this.embedding.getEmbedding(question);

    console.log(`🔎 Searching ChromaDB...`);
    const results = await this.chroma.query(
      this.collectionId,
      questionEmbedding,
      3
    );
    
    if (!results.documents || results.documents[0].length === 0) {
      console.log('❌ No relevant documents found');
      return null;
    }

    // Use qwen2.5-coder for generating the answer
    const answer = await this.askOllama(question, results.documents[0]);
    
    return {
      question,
      contexts: results.documents[0],
      answer,
    };
  }

  async askOllama(question, contexts) {
    const prompt = `Use the following context to answer the question. If you can't find the answer, say so.

Context:
${contexts.join('\n\n')}

Question: ${question}

Answer:`;

    console.log(`🤔 Asking ${this.llmModel} to generate answer...`);
    
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.llmModel,  // ← USING YOUR MODEL: qwen2.5-coder:1.5b
        prompt: prompt,
        stream: false,
      }),
    });

    const data = await response.json();
    return data.response;
  }

  async countDocuments() {
    return this.chroma.countRecords(this.collectionId);
  }
}

// Test it
async function main() {
  const rag = new RAGApplication();
  
  console.log('🚀 Starting RAG with qwen2.5-coder:1.5b...\n');
  await rag.initialize();
  
  const count = await rag.countDocuments();
  console.log(`📊 Current documents: ${count}`);

  if (count === 0) {
    console.log('\n📝 Adding sample documents...');
    await rag.addDocuments([
      {
        text: 'ChromaDB is a vector database for AI applications. It stores embeddings and enables semantic search.',
        metadata: { topic: 'database' }
      },
      {
        text: 'Vector databases understand meaning, not just keywords. They can find "car" when you search "automobile".',
        metadata: { topic: 'semantic' }
      },
      {
        text: 'RAG (Retrieval-Augmented Generation) combines vector search with LLMs for better answers.',
        metadata: { topic: 'rag' }
      }
    ]);
  }

  // Test query
  const question = 'What is a vector database and how does it understand meaning?';
  console.log(`\n❓ Question: ${question}`);
  
  const result = await rag.query(question);
  if (result) {
    console.log(`\n💡 Answer: ${result.answer}`);
  }
}

main().catch(console.error);