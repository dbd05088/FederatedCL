import argparse
import json
import os

import openai
from openai import OpenAI
import time
import copy
import base64

NUM_SECONDS_TO_SLEEP = 0.5

client = OpenAI(
    # This is the default and can be omitted
    # api_key=os.environ.get("OPENAI_API_KEY"),
    api_key="sk-proj-uw3bhK9gbQ9hri0UHhjKT3BlbkFJrQgQPShJ3kDn639B7OI9"
)

def get_eval(content: str, max_tokens: int):
    user_content = [
                        {"type":"text", "text":content['text']},
                    ]
    for base64_image in content['image']:
        user_content.append({"type": "image_url",
                             "image_url":{"url": f"data:image/jpeg;base64,{base64_image}"}}
                        )

    while True:
        try:
            '''
            
            '''
            response = client.chat.completions.create(
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                },{
                    'role': 'user',
                    'content': user_content
                }],
                model='gpt-4o',
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return response.choices[0].message.content

def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 3:
            return [float(sp[0]), float(sp[1]), float(sp[2])]
        else:
            print('error', review)
            return [-1, -1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1, -1]

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-r', '--result')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    print(f"start evaluating {args.result}")
    results_dict = json.load(open(args.result, 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        output_dir = os.path.dirname(args.output)
        os.makedirs(output_dir, exist_ok=True)
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    rule = {"role": "Assistant", 
            "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above.\
                The user asks the question on observing an image or mutiple images. The images replaces <image> tokens in [Question].\n\
                Please rate the relevance, accuracy, level of details of the response compared to ground-truth answer. \
                The assistant receives three scores on a scale of 1 to 10, where a higher score indicates better overall performance. The first score indicates the relevancy of the assistant's response to the image and question. The second score indicates the accuracy of the asssitant's response to the question and ground-truth answer. The final score indicates the overall score of the response.\n\
                Please first output a single line containing only three values indicating the scores for Assistant. The three scores are separated by a space.\n\
                In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."}
                #For your reference, the visual content in the image is represented with a few sentences describing the image.
    handles = []
    for item in results_dict:
        review_item = copy.deepcopy(item)
        prompt = rule['prompt']
        role = rule['role']
        image_paths = item['image_file']
        base64_imgs = []
        if isinstance(image_paths, list):
            for img_path in image_paths:
                base64_imgs.append(encode_image(img_path))
        else:
            base64_imgs.append(encode_image(image_paths))
        content = {
            "text": (f'[Question]\n{item["input"]}\n\n'
                   f'[Answer]\n{item["gt_sentence"]}\n\n[End of Answer]\n\n'
                   f'[{role}]\n{item["sentence"]}\n\n[End of {role}]\n\n'
                   f'[System]\n{prompt}\n\n'),
            "image":
                   base64_imgs
                   }
        

        review = get_eval(content, args.max_tokens)
        scores = parse_score(review)
        review_item['content'] = review
        review_item['score'] = scores
        cur_reviews.append(review_item)
        review_file.write(json.dumps(review_item) + '\n')
        review_file.flush()
    review_file.close()
    
