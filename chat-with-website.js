import { Readability } from "@mozilla/readability";
import axios from "axios";
import "dotenv/config";
import { XMLParser } from "fast-xml-parser";
import { JSDOM } from "jsdom";
import { LLMChain, RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { Document } from "langchain/document";
import { BaseDocumentLoader } from "langchain/document_loaders/base";
import { HtmlToTextTransformer } from "langchain/document_transformers/html_to_text";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";
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

class SitemapLoader extends BaseDocumentLoader {
    url;

    constructor(url) {
        super();
        this.url = url;
    }

    async load() {
        const response = await axios.post(this.url);
        const xml = response.data;
        const parser = new XMLParser();
        const results = parser.parse(xml);
        const documents = [];
        for (const url of results.urlset.url) {
            console.log(`Loading ${url.loc} data... `);
            try {
                const response = await axios.get(url.loc);
                const html = response.data;
                const doc = new JSDOM(html, {
                    url: url.loc,
                });
                const reader = new Readability(doc.window.document);
                const article = reader.parse();
                const pageContent = article.textContent;
                const metadata = {
                    type: "page",
                    title: article.title,
                    length: pageContent.length,
                    siteName: article.siteName,
                    author: article.byline,
                    lang: article.lang,
                    url: url.loc,
                    date: url.lastmod,
                };
                const document = new Document({ pageContent, metadata });
                documents.push(document);
                console.log(`Done!\n`);
            } catch (e) {
                console.log(`URL not found. Skipped!\n`);
            }
        }
        return documents;
    }
}

const finish = async () => {
    prompt.close();
    await client.disconnect();
    process.exit();
};

async function main() {
    const xmlLoader = new SitemapLoader("https://niara.ai/page-sitemap.xml");

    const docs = await xmlLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    const transformer = new HtmlToTextTransformer();

    const sequence = splitter.pipe(transformer);

    const newDocs = await sequence.invoke(docs);

    const docsMapped = newDocs.map((doc) => {
        return {
            ...doc,
            pageContent: doc.pageContent,
        };
    });

    const vectorStore = await RedisVectorStore.fromDocuments(
        docsMapped,
        new OpenAIEmbeddings(),
        {
            redisClient: client,
            indexName: "chat-with-website",
        }
    );

    /* Usage as part of a chain */
    const model = new ChatOpenAI({
        model: "gpt-3.5-turbo",
        temperature: 0,
        prefixMessages: [
            {
                role: "system",
                content: `Você é uma assistente de IA. Apenas responda as perguntas relativas ao documento. Caso você não saiba a resposta, responda "Não sei".`,
            },
        ],
    });
    const chat = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(1), {
        returnSourceDocuments: true,
    });

    prompt.write("❓ Ask anything about my website!\n");
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
            const contextChain = await chat.call({
                query: question,
                // verbose: true,
            });
            const context = contextChain.text;
            const sources = contextChain.sourceDocuments.map(
                (source) =>
                    `\n- [${source.metadata.title}](${source.metadata.url})`
            );

            const responseTemplate = `Você é uma assistente de IA que responde perguntas sobre um website. Dado o contexto e fonte abaixo, responda à duvida do usuário. Caso você não saiba, responda 'Não sei'.\n"
            Contexto: {context}
            Fonte: {sources}
            Pergunta: {question}`;
            const responsePromptTemplate = new PromptTemplate({
                template: responseTemplate,
                inputVariables: ["context", "sources", "question"],
            });
            const responseChain = new LLMChain({
                llm: new OpenAI({ model: "gpt-4", temperature: 0.7 }),
                prompt: responsePromptTemplate,
                // verbose: true,
            });

            const response = await responseChain.call({
                context,
                sources,
                question,
            });

            console.log(`${response.text}\n`);
            console.log(
                `Fonte: ${contextChain.sourceDocuments.map(
                    (source) =>
                        `\n- [${source.metadata.title}](${source.metadata.url})`
                )}\n`
            );
            loopQuestion();
        });
    };

    loopQuestion();
}

main();
