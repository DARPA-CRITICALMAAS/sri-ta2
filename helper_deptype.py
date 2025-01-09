import pandas

class new:
    def __init__(self,llm,params):
        self.params=params
        self.llm=llm
        #Compile options
        try:
            taxonomy=pandas.read_csv(params.taxonomy)
        except:
            taxonomy=pandas.read_csv(params.taxonomy,encoding='latin1')
        
        options=list(taxonomy['Deposit type'])
        descriptions=list(taxonomy['Description'])
        descriptions=[x if type(x)==str else '' for x in descriptions]
        
        
        self.options=options
        self.descriptions=descriptions
    
    
    def summary(self,text,options=None,descriptions=None):
        if options is None:
            options=self.options
        
        if descriptions is None:
            descriptions=self.descriptions
        
        chunks=self.llm.chunk(text,L=int((self.params.lm_context_window-20000)*0.8))
        list_of_options=''.join(['a%03d. %s. %s\n'%(i+1,options[i],descriptions[i]) for i in range(len(options))])
        
        system="You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context description and PDF report and answer the question about the PDF report."
        user="Context: NI 43-101 applies broadly to companies both public and private, and to a variety of disclosures types including mineral exploration reports, reporting of resources and reserves, presentations, oral comments, and websites. NI 43-101 covers mineral products such, precious metals and solid energy commodities as well as bulk minerals, dimension stone, precious stones and mineral sands commodities. The following is an NI 43-101 report describing a mineral resource. Please read and answer the question below.\n\n```report\n{text}\n```\nQuestion: Which of the following mineral deposit types best fits the area that the context PDF report describes? Options:\n{list_of_options}\n\nPlease summarize the original PDF reports on its descriptions of the likely deposit types in the area, and select the mineral deposit type that best fits the area." #text, list_of_options
        
        summaries=[self.llm.qa(system,user.format(text=chunk,list_of_options=list_of_options)) for chunk in chunks]
        summaries=''.join(['Section %d: %s\n'%(i,s) for (i,s) in enumerate(summaries)])
        
        return summaries
    
    def classify(self,text,options=None,descriptions=None):
        if options is None:
            options=self.options
        
        options=options+['Irrelevant question.']
        if descriptions is None:
            descriptions=self.descriptions
        
        descriptions=descriptions+['Not a mineral site.']
        
        text=self.llm.chunk(text,L=int((self.params.lm_context_window-20000)*0.8))[0]
        list_of_options=''.join(['a%03d. %s. %s\n'%(i+1,options[i],descriptions[i]) for i in range(len(options))])
        
        system="You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context description and PDF report and answer the multiple-choice question about the PDF report."
        
        user="Context: NI 43-101 applies broadly to companies both public and private, and to a variety of disclosures types including mineral exploration reports, reporting of resources and reserves, presentations, oral comments, and websites. NI 43-101 covers mineral products such, precious metals and solid energy commodities as well as bulk minerals, dimension stone, precious stones and mineral sands commodities. The following is an NI 43-101 report describing a mineral resource. Please read and answer the question below.\n\n```report\n{text}\n```\nQuestion: Which of the following mineral deposit types best fits the area that the context PDF report describes? Options:\n{list_of_options}\n\nPlease select the mineral deposit type that best fits the area that the context PDF report describes. Please choose only 1 most likely option. Answer the question with only the 4-letter alpha-numeric id (a***) of the most likely option and nothing else." #text, list_of_options
        
        logp,_=self.llm.multiple_choice(system,user.format(text=text,list_of_options=list_of_options),len(options))
        return logp
    
    def explain(self,text,options=None,descriptions=None):
        if options is None:
            options=self.options
        
        if descriptions is None:
            descriptions=self.descriptions
        
        text=self.llm.chunk(text,L=int((self.params.lm_context_window-20000)*0.8))[0]
        list_of_options=''.join(['a%03d. %s. %s\n'%(i+1,options[i],descriptions[i]) for i in range(len(options))])
        
        system="You are a helpful assistant specialized in reviewing geological publications and answering questions about them. Please read the context description and PDF report and answer the question about the PDF report."
        
        user="Context: NI 43-101 applies broadly to companies both public and private, and to a variety of disclosures types including mineral exploration reports, reporting of resources and reserves, presentations, oral comments, and websites. NI 43-101 covers mineral products such, precious metals and solid energy commodities as well as bulk minerals, dimension stone, precious stones and mineral sands commodities. The following is an NI 43-101 report describing a mineral resource. Please read and answer the question below.\n\n```report\n{text}\n```\nQuestion: Which of the following mineral deposit types best fits the area that the context PDF report describes? Options:\n{list_of_options}\n\nPlease select the mineral deposit type that best fits the area that the context PDF report describes. Please choose only 1 most likely option. Answer the question with the most likely option and explain your reasoning."
        
        explanation=self.llm.qa(system,user.format(text=text,list_of_options=list_of_options))
        return explanation
    
    def run(self,text):
        chunks=self.llm.chunk(text,L=int((self.params.lm_context_window-20000)*0.8))
        if len(chunks)>1:
            text=self.summary(text)
        
        logp=self.classify(text)
        explanation=self.explain(text)
        return logp,explanation
        