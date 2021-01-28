from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagConfig
from transformers import BartForConditionalGeneration, BartConfig

config = RagConfig.from_pretrained("facebook/rag-sequence-nq")
#config.generator = BartConfig.from_pretrained("facebook/bart-base")

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq") 
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever,\
generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base"), config=config) 


model.config.n_docs = 4
retriever.config.n_docs = 4


input_dict = tokenizer.prepare_seq2seq_batch("Where were the spiders?", return_tensors="pt") 
generated = model.generate(input_ids=input_dict["input_ids"], extra_context="Spiders killed my mother.") 
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 

# should give 54 => google says either 44 or 51