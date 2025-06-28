from litellm import completion 
# from pydantic import BaseModel
# from enum import Enum 
import base64
from colorama import Fore 
import json
import os 

# class Options(Enum): 
#     doji = '1-doji'
#     hammer = '2-hammer' 
#     hanging_man = '3-hanging man' 
#     bull_engulfing = '4-bullish engulfing' 
#     bear_engulfing = '5-bearish engulfing' 
#     morning_star = '6-morning star' 
#     evening_star = '7-evening star' 

# class Candle(BaseModel):
#     result: Options

os.environ["OPENAI_API_KEY"] = ""

def llm_call(base64_image) -> dict:
    stream = completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content":[
                    {
                        "type":"text", 
                        "text":"""What type of candlestick pattern is shown in the last 3-5 candles? 
                                Pick one out of Doji, Hammer, Hanging Man, Bull Engulfing, Bear Engulfing, Morning Star or Evening Star.
                                Return a json object with the classification as one of the eight options: 
                                    doji = '1-doji'
                                    hammer = '2-hammer' 
                                    hanging_man = '3-hanging man' 
                                    bull_engulfing = '4-bullish engulfing' 
                                    bear_engulfing = '5-bearish engulfing' 
                                    morning_star = '6-morning star' 
                                    evening_star = '7-evening star' 
                                """
                    },
                    {
                        "type":"image_url", 
                        "image_url":{
                            "url":f"data:image/png;base64,{base64_image}"
                        }
                    },
                    ]
            }
        ],
        stream=True,
        response_format={"type": "json_object"},
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)


if __name__ == "__main__": 
    image_file_name = '0cb3c8e0-48a2-11f0-93ad-83dd4b7f9dbc.png'
    with open(f'data/{image_file_name}', 'rb') as image_file: 
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        res = llm_call(encoded_string)
        print(Fore.LIGHTYELLOW_EX + str(res) + Fore.RESET) 

        with open('openaires.txt', 'a', newline='') as f: 
            f.write(f"{image_file_name}, {res}")
