import { ChromaClient } from './chroma-client.js';

async function testChroma() {
  const client = new ChromaClient('http://localhost:8000');
  
  try {
    console.log('🔍 Testing ChromaDB...\n');
    
    // Test basic connection
    console.log('1️⃣ Testing connection:');
    const version = await client.version();
    console.log('✅ Version:', version);
    
    const heartbeat = await client.heartbeat();
    console.log('✅ Heartbeat received');

    // Create a unique collection name with timestamp
    const collectionName = `test_${Date.now()}`;
    console.log(`\n2️⃣ Creating collection: ${collectionName}`);
    
    const collection = await client.getOrCreateCollection(collectionName, {
      space: 'cosine',
      description: 'Test collection'
    });
    console.log('✅ Collection created with ID:', collection.id);

    // Add a document
    console.log('\n3️⃣ Adding document:');
    await client.addDocuments(
      collection.id,
      ['This is a test document about ChromaDB'],
      ['doc1']
    );
    console.log('✅ Document added');

    // Count documents
    const count = await client.countRecords(collection.id);
    console.log(`\n4️⃣ Document count: ${count}`);

    // Query
    console.log('\n5️⃣ Testing query:');
    const results = await client.query(
      collection.id,
      [0.1, 0.2, 0.3, 0.4],
      1
    );
    
    if (results.documents && results.documents[0]) {
      console.log('✅ Query results:');
      console.log('   Found:', results.documents[0][0]);
    }

    console.log('\n✅ All tests passed! ChromaDB is working!');

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

testChroma();