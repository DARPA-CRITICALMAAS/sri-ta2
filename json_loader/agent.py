
import backoff 
import openai
openai_api_key = # your openai key
client = openai.OpenAI(
  api_key=openai_api_key,
)
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


import os
import math
import pandas
import torch

def is_done(response):
    done=response.find('Done.')>=0
    return done

def get_code(response):
    s=response.find('```python')
    if s<0:
        return None
    
    response=response[s+len('```python'):]
    e=response.find('```')
    if e<0:
        return None
    
    return response[:e]

#Provide a python runtime
import io
from contextlib import redirect_stdout
import json
import sys

def eval_f(code):
    df_out=pandas.DataFrame()
    f_cap = io.StringIO()
    err_msg='N/A'
    df_out_vis='N/A'
    
    with redirect_stdout(f_cap):
        try:
            d={}
            exec(code,d)
            df_out=d['entrypoint']()
            if len(df_out)>0:
                df_out_vis=repr(df_out)[:2000]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            err_msg=repr(e)[:2000]
            #err_msg=repr((e,exc_type,'line number',exc_tb.tb_lineno))[:1000]
    
    console = f_cap.getvalue()
    console=console[:2000]
    return df_out,df_out_vis,err_msg,console


def run_df_agent(request,max_iter=100):
    msg=[]
    msg.append({"role": "system", "content": "You are a helpful, honest, creative, persistent agent specialized in writing and debugging python scripts for converting pandas data frames following user specifications. Please read the following API definitions and design a python function to execute the user's request.\n\nYou have access to a python environment equipped with libraries such as pandas, torch, numpy, math, scipy and common string processing libraries. Please write a function in the form of\n```python\nimport pandas\nimport ... #Please explicitly import all libraries you need\ndef entrypoint():\n\t...\n\treturn df_out\n```\nYou must use `entrypoint` as the name of your function. The `entrypoint` function must return a valid pandas data frame as the output. For the clarity of the results, please keep only columns that are relevant to the user's request to reduce the size of the output data frame. Functions you wrote in your response will be evaluted automatically. Please write only function definitions and not scripts. For debugging, you will be informed of the first 2000 characters of the returned df_out data frame, exception messages and the first 2000 characters of the console print() output. Please do not use any simplifying assumptions. Please reduce hardcoding whenever possible. "})
    msg.append({"role": "user", "content": 'Could you please develop a python script for the request below:\n"%s"'%request})
    
    log=[]
    df_out=None
    for iter in range(100):
        response=completions_with_backoff(model="gpt-4-turbo", messages=msg).choices[0].message.content
        print('iter %d\n\n'%iter)
        print(response)
        code=get_code(response)
        done=is_done(response)
        if done and code is None:
            log.append({'msg':msg,'response':response,'code':code,'df_out':df_out,'first_line':'','err_msg':'','console':''})
            torch.save(log,'log/log.pt')
            break;
        elif not done and code is None:
            msg.append({"role": "assistant", "content": response})
            msg.append({"role": "user", "content": 'Please reply to my previous question.'})
            log.append({'msg':msg,'response':response,'code':code,'df_out':df_out,'first_line':first_line,'err_msg':err_msg,'console':console})
        else:
            df_out,first_line,err_msg,console=eval_f(code) 
            try:
                df_out.to_csv('log/out_iter%03d.csv'%iter)
            except:
                pass
            
            prompt_i="Output data frame:\n%s\nError message:\n%s\nConsole output:\n%s"%(first_line,err_msg,console)
            prompt_i+='\nDoes the output data frame satisfy the user request? If not, please explain where the issues are, and rewrite your program. If the output does satisfy the user request, please explain the exhaustiveness of user request coverage and then say "Done.". '
            msg.append({"role": "assistant", "content": response})
            msg.append({"role": "user", "content": prompt_i})
            log.append({'msg':msg,'response':response,'code':code,'df_out':df_out,'first_line':first_line,'err_msg':err_msg,'console':console})
            torch.save(log,'log/log.pt')
            print(prompt_i)
   
    return df_out
