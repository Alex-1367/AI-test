import fs from 'fs';

const MODEL_PATH = '/usr/share/ollama/.ollama/models/blobs/sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104';

console.log('🔬 RAW BYTE DUMP AROUND METADATA END');
console.log('=' .repeat(70));

try {
    const buffer = fs.readFileSync(MODEL_PATH);
    const fileSize = buffer.length;
    
    // The metadata end offset from your run
    const metadataEnd = 5931783;
    
    console.log(`📊 File size: ${fileSize} bytes`);
    console.log(`📍 Metadata ends at: ${metadataEnd} (0x${metadataEnd.toString(16)})`);
    
    // Dump 128 bytes before and after metadataEnd
    const dumpStart = Math.max(0, metadataEnd - 64);
    const dumpEnd = Math.min(fileSize, metadataEnd + 128);
    
    console.log(`\n📝 Raw bytes from 0x${dumpStart.toString(16)} to 0x${dumpEnd.toString(16)}:`);
    console.log('-'.repeat(80));
    
    for (let i = dumpStart; i < dumpEnd; i += 16) {
        const line = [];
        const hex = [];
        const ascii = [];
        
        for (let j = 0; j < 16 && i + j < dumpEnd; j++) {
            const byte = buffer[i + j];
            hex.push(byte.toString(16).padStart(2, '0'));
            ascii.push(byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : '.');
        }
        
        const offset_str = i.toString(16).padStart(8, '0');
        const hex_str = hex.join(' ').padEnd(48);
        const ascii_str = ascii.join('');
        
        // Mark the metadata end point
        const marker = (i <= metadataEnd && i + 16 > metadataEnd) ? ' <--- METADATA ENDS HERE' : '';
        console.log(`${offset_str}: ${hex_str} | ${ascii_str}${marker}`);
    }
    
    // Also look for the string "token_embd.weight" in the file
    console.log('\n🔍 Searching for "token_embd.weight" in file...');
    
    const searchString = 'token_embd.weight';
    const searchBuffer = Buffer.from(searchString);
    
    let foundCount = 0;
    for (let i = 0; i < fileSize - searchBuffer.length; i++) {
        let found = true;
        for (let j = 0; j < searchBuffer.length; j++) {
            if (buffer[i + j] !== searchBuffer[j]) {
                found = false;
                break;
            }
        }
        if (found) {
            console.log(`✅ Found at offset: ${i} (0x${i.toString(16)})`);
            foundCount++;
            
            // Show context around this occurrence
            const contextStart = Math.max(0, i - 32);
            const contextEnd = Math.min(fileSize, i + 64);
            
            console.log(`   Context around offset ${i}:`);
            for (let k = contextStart; k < contextEnd; k += 16) {
                const ctxHex = [];
                const ctxAscii = [];
                for (let l = 0; l < 16 && k + l < contextEnd; l++) {
                    const byte = buffer[k + l];
                    ctxHex.push(byte.toString(16).padStart(2, '0'));
                    ctxAscii.push(byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : '.');
                }
                const marker2 = (k <= i && k + 16 > i) ? ' <--- HERE' : '';
                console.log(`   ${k.toString(16).padStart(8, '0')}: ${ctxHex.join(' ').padEnd(48)} | ${ctxAscii.join('')}${marker2}`);
            }
            console.log('');
        }
    }
    
    console.log(`\n📊 Found "${searchString}" ${foundCount} times in file`);
    
    // Now let's look at the first few occurrences of what might be tensor names
    console.log('\n🔍 Looking for potential tensor names (strings starting with "blk." or "token"):');
    
    const patterns = ['blk.', 'token', 'output', 'norm', 'attn'];
    const found = [];
    
    for (let i = 0; i < fileSize - 10; i++) {
        for (const pattern of patterns) {
            if (buffer[i] === pattern.charCodeAt(0)) {
                let match = true;
                for (let j = 0; j < pattern.length; j++) {
                    if (buffer[i + j] !== pattern.charCodeAt(j)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    // Read up to 30 chars or until null
                    let name = '';
                    for (let j = 0; j < 30 && i + j < fileSize; j++) {
                        const c = buffer[i + j];
                        if (c === 0) break;
                        name += String.fromCharCode(c);
                    }
                    if (!found.some(f => Math.abs(f.offset - i) < 10)) {
                        found.push({ offset: i, name });
                        console.log(`   Found "${name}" at offset ${i} (0x${i.toString(16)})`);
                    }
                    i += name.length;
                }
            }
        }
    }

} catch (error) {
    console.error('❌ Error:', error);
}