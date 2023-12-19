import "dotenv/config";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RedisVectorStore } from "langchain/vectorstores/redis";
import readline from "readline";
import { createClient } from "redis";

const prompt = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const client = createClient({
    url: process.env.REDIS_URL ?? "redis://localhost:6379",
});
await client.connect();

const finish = async () => {
    prompt.close();
    await client.disconnect();
    process.exit();
};

async function main() {
    const loader = new PDFLoader("files/Niara-Pitch-Deck-Julho-2023.pdf");

    const docs = await loader.loadAndSplit();
    const docsMapped = docs.map((doc) => ({
        ...doc,
        pageContent: doc.pageContent,
    }));

    const vectorStore = await RedisVectorStore.fromDocuments(
        docsMapped,
        new OpenAIEmbeddings(),
        {
            redisClient: client,
            indexName: "docs",
        }
    );

    /* Usage as part of a chain */
    const model = new ChatOpenAI({
        model: "gpt-3.5-turbo",
        temperature: 0.7,
        prefixMessages: [
            {
                role: "system",
                content: `Você é uma assistente de IA. Apenas responda as perguntas relativas ao documento em questão. Caso você não saiba a resposta, responda "Não sei".`,
            },
        ],
    });
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(4), {
        returnSourceDocuments: true,
    });

    prompt.write(
        "🤖 Bem-vindo ao Niara CLI! Pergunte qualquer coisa sobre a empresa!\n"
    );
    const loopQuestion = () => {
        prompt.question("> ", async (question) => {
            if (
                question === "exit" ||
                question === "quit" ||
                question === "sair" ||
                question === "q"
            ) {
                await finish();
            }
            const chainRes = await chain.call({
                query: question,
                // verbose: true,
            });
            // console.log(chainRes.sourceDocuments);
            console.log(`R: ${chainRes.text}\n`);
            loopQuestion();
        });
    };

    loopQuestion();
}

main();
