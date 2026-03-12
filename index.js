import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';


// Load the PDF document

async function indexDocument() {
    const PDF_PATH = '\dsa.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();

    console.log("PDF document loaded successfully.");

    // Chunking the document into smaller pieces

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, // Adjust chunk size as needed
        chunkOverlap: 200, // Adjust overlap as needed
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

    //console.log(`${chunkedDocs.length} chunks created from the document.`);
    //console.log(rawDocs.length, 'raw documents loaded.');

    console.log("Document chunking completed successfully.");

    // Vector embedding model (convert text to vectors)
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });

    console.log("Embeddings model initialized successfully.");

    // Configure Pinecone client / Database
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    console.log("Pinecone client initialized successfully.");

    // Langchain ( Bring : chunking, embedding, and database indexing )
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5
    });

    console.log("Document indexed successfully in Pinecone.");

}
// Call the function (if needed in an async environment, use await indexDocument())
indexDocument();
