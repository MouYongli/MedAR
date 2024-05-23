import transformers
import torch
torch.cuda.empty_cache()
#OpenBioLLM-70B
model_id = "aaditya/OpenBioLLM-Llama3-70B"
#model_id = "aaditya/OpenBioLLM-Llama3-8B"
#os.environ[“CUDA_VISIBLE_DEVICES”]=“0,1,2,3"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cpu",
)

messages = [
    {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. as a substantial part of our concept of a minimally invasive cochlear implant ci surgery we developed an automated IS tool studies on an artificial ST model were performed in order to evaluate force application when using the insertion tool"},
    {"role": "user", "content": "What is ST in the sentence?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
#An artificial scala tympani (ST) model is a simulated or manufactured model that mimics the scala tympani, 
#which is one of the three chambers in the cochlea of the inner ear. 
#This model is used for research purposes, 
#such as evaluating the insertion force of cochlear implant electrodes during surgical procedures. 
#It allows researchers to test and refine the insertion techniques and tools without the need for 
#human subjects.


'''#ClinicalBERT
from transformers import AutoTokenizer, AutoModel, DistilBertForMaskedLM, pipeline
#tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

#model = DistilBertForMaskedLM.from_pretrained("medicalai/ClinicalBERT")
unmasker = pipeline('fill-mask', model='medicalai/ClinicalBERT')

text = "as a substantial part of our concept of a minimally invasive cochlear implant ci surgery we developed an automated IS tool studies on an artificial [MASK] model were performed in order to evaluate force application when using the insertion tool."
print(unmasker(text))

#life
#muscle
#pressure

#DistilBert The current model class (DistilBertModel) is not compatible with `.generate()`, as it doesn't have a language model head.
'''
'''
#PubMedBERT
from transformers import AutoTokenizer, AutoModel, DistilBertForMaskedLM, pipeline

unmasker = pipeline('fill-mask', model='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')

text = "as a substantial part of our concept of a minimally invasive cochlear implant ci surgery we developed an automated IS tool studies on an artificial [MASK] model were performed in order to evaluate force application when using the insertion tool."
print(unmasker(text))

#ear
#cochlea
#head
#larynx
'''


