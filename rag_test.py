from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagConfig

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq") 
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).to("cuda:0")
model.add_tokens()

model.config.n_docs = 6
retriever.config.n_docs = 6

model.config.n_docs_splits = 3
retriever.config.n_docs_splits = 3

model.skip_ec = True
model.skip_ec = True

input_dict = tokenizer.prepare_seq2seq_batch("am i a cool person", return_tensors="pt").to("cuda:0")
generated = model.generate(**input_dict, extra_context=["Cats are cool animals!"], num_beams=4) 
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 

# should give 54 => google says either 44 or 51