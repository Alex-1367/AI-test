import { ChromaClient } from './chroma-client.js';

async function simpleTest() {
  const client = new ChromaClient('http://localhost:8000');
  
  try {
    console.log('Simple ChromaDB v2 test');
    console.log('=======================');
    
    // Just test basic connectivity
    const version = await client.version();
    console.log('✅ Connected to ChromaDB version:', version);
    
    const heartbeat = await client.heartbeat();
    console.log('✅ Heartbeat received');
    
  } catch (error) {
    console.error('❌ Connection failed:', error.message);
  }
}

simpleTest();