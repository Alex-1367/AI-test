// Debug-oriented system showing vectors and embeddings

import { ChromaClient } from './chroma-client.js';
import { createInterface } from 'readline';

class CleanRAG {
  constructor() {
    this.chroma = new ChromaClient('http://localhost:8000');
    this.ollamaUrl = 'http://localhost:11434';
    this.model = 'qwen2.5-coder:1.5b';
    this.collectionId = null;
  }

  // Helper to format vector preview
  formatVectorPreview(vector, showCount = 10) {
    if (!vector || !vector.length) return 'No vector';
    
    const first = vector.slice(0, showCount).map(n => n.toFixed(4)).join(', ');
    const last = vector.slice(-showCount).map(n => n.toFixed(4)).join(', ');
    
    return {
      dimensions: vector.length,
      vector: `[${first} ... ${last}]`
    };
  }

  async log(msg, data = null, important = false) {
    const line = important ? '🔴' : '─'.repeat(40);
    console.log(`\n${line}`);
    console.log(`🔍 ${msg}`);
    if (data) {
      if (typeof data === 'object') {
        console.log(JSON.stringify(data, null, 2));
      } else {
        console.log(data);
      }
    }
  }

  async callOllamaEmbedding(text) {
    await this.log(`📤 Ollama Embedding API Call`, {
      model: this.model,
      text: text.length > 50 ? text.substring(0, 50) + '...' : text
    });

    const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        prompt: text,
      }),
    });

    const data = await response.json();
    const preview = this.formatVectorPreview(data.embedding);
    
    await this.log(`📥 Ollama Returns Vector`, {
      dimensions: preview.dimensions,
      full_sample: preview.vector
    }, true); // Important = true (red line)

    return data.embedding;
  }

  async callOllamaGenerate(prompt, context) {
    await this.log(`📤 Ollama Generate API Call`, {
      model: this.model,
      promptLength: prompt.length,
      contextCount: context.length
    });

    const response = await fetch(`${this.ollamaUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        prompt: prompt,
        stream: false,
      }),
    });

    const data = await response.json();
    
    await this.log(`📥 Ollama Generate Response`, {
      responsePreview: data.response.substring(0, 200) + (data.response.length > 200 ? '...' : ''),
      fullResponse: data.response
    });

    return data.response;
  }

  async initialize() {
    await this.log('🚀 Initializing ChromaDB Collection');
    
    const collection = await this.chroma.getOrCreateCollection('clean_test', {
      space: 'cosine',
      description: 'Clean test collection'
    });
    
    this.collectionId = collection.id;
    
    await this.log('✅ Collection Ready', {
      collectionId: this.collectionId,
      name: collection.name
    });
  }

  async addDocument(text, metadata = {}) {
    await this.log(`📄 Adding Document: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);

    // Get embedding from Ollama
    const embedding = await this.callOllamaEmbedding(text);
    
    // Send to ChromaDB
    await this.log(`📤 Storing in ChromaDB...`);
    
    const result = await this.chroma.addDocuments(
      this.collectionId,
      [text],
      [`doc_${Date.now()}`],
      [embedding],
      [metadata]
    );

    await this.log(`✅ Document Stored in ChromaDB`);
  }

  async search(query) {
    await this.log(`🔎 Searching for: "${query}"`);

    // Convert query to embedding
    const queryEmbedding = await this.callOllamaEmbedding(query);
    
    // Search in ChromaDB
    await this.log(`📤 Querying ChromaDB...`);
    
    const results = await this.chroma.query(
      this.collectionId,
      queryEmbedding,
      3
    );

    // Show results
    await this.log(`📥 ChromaDB Search Results`, {
      found: results.ids[0].length,
      matches: results.ids[0].map((id, i) => ({
        id: id,
        document: results.documents[0][i].substring(0, 100) + 
                 (results.documents[0][i].length > 100 ? '...' : ''),
        similarity: results.distances ? (1 - results.distances[0][i]).toFixed(4) : 'N/A',
        distance: results.distances ? results.distances[0][i].toFixed(4) : 'N/A'
      }))
    }, true); // Important = true

    return results;
  }

  async listAllDocuments() {
    await this.log(`📚 Listing All Documents`);
    
    const records = await this.chroma.getRecords(this.collectionId, {
      include: ["documents", "metadatas", "embeddings"],
      limit: 100
    });

    if (records.ids.length === 0) {
      await this.log(`📭 No documents in collection`);
      return;
    }

    await this.log(`📚 Documents in Collection (${records.ids.length} total)`, {
      documents: records.ids.map((id, i) => ({
        id: id,
        text: records.documents[i].substring(0, 100) + 
              (records.documents[i].length > 100 ? '...' : ''),
        metadata: records.metadatas[i],
        vector: records.embeddings ? {
          dimensions: records.embeddings[i].length,
          first10: records.embeddings[i].slice(0, 10).map(n => n.toFixed(4)),
          last10: records.embeddings[i].slice(-10).map(n => n.toFixed(4))
        } : 'Not retrieved'
      }))
    }, true); // Important = true
  }

  async showStats() {
    const count = await this.chroma.countRecords(this.collectionId);
    await this.log(`📊 Collection Statistics`, {
      documentCount: count,
      collectionId: this.collectionId
    });
  }
}

// Interactive console
async function main() {
  const rag = new CleanRAG();
  
  console.log('\n' + '🚀'.repeat(20));
  console.log('🚀 CLEAN RAG TEST - Vector preview (first/last 10)');
  console.log('🚀'.repeat(20) + '\n');

  await rag.initialize();
  await rag.showStats();

  const readline = createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const showHelp = () => {
    console.log('\n📋 Commands:');
    console.log('  add <text>     - Add document (shows vector preview)');
    console.log('  search <query> - Search (shows vector preview + results)');
    console.log('  list           - List all documents with vector previews');
    console.log('  stats          - Show document count');
    console.log('  clear          - Clear screen');
    console.log('  quit           - Exit');
    console.log('─'.repeat(40));
  };

  showHelp();

  const ask = () => {
    readline.question('\n❯ ', async (input) => {
      const [command, ...args] = input.trim().split(' ');
      const text = args.join(' ');

      switch(command) {
        case 'add':
          if (!text) console.log('❌ Please provide text');
          else await rag.addDocument(text);
          break;

        case 'search':
          if (!text) console.log('❌ Please provide search query');
          else await rag.search(text);
          break;

        case 'list':
          await rag.listAllDocuments();
          break;

        case 'stats':
          await rag.showStats();
          break;

        case 'clear':
          console.clear();
          showHelp();
          break;

        case 'quit':
        case 'exit':
          readline.close();
          return;

        default:
          console.log('❌ Unknown command');
          showHelp();
      }

      ask();
    });
  };

  ask();
}

main().catch(console.error);