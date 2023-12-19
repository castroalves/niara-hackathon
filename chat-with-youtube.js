import "dotenv/config";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
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
    prompt.write("▶️ Chat with YouTube!\n");

    const loader = YoutubeLoader.createFromUrl(
        "https://www.youtube.com/watch?v=X280T0lDozU",
        {
            language: "en",
            addVideoInfo: true,
        }
    );

    const docs = await loader.loadAndSplit(
        new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 100,
        })
    );
    const docsMapped = docs.map((doc) => ({
        ...doc,
        pageContent: doc.pageContent,
    }));

    const vectorStore = await RedisVectorStore.fromDocuments(
        docsMapped,
        new OpenAIEmbeddings(),
        {
            redisClient: client,
            indexName: "chat-with-youtube",
        }
    );

    /* Usage as part of a chain */
    const model = new ChatOpenAI({
        model: "gpt-3.5-turbo",
        temperature: 0.7,
        prefixMessages: [
            {
                role: "system",
                content: `Você é uma assistente de IA. Apenas responda as perguntas relativas à transcrição do vídeo do YouTube. Caso você não saiba a resposta, responda "Não sei".`,
            },
        ],
    });
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(4), {
        returnSourceDocuments: true,
    });
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
