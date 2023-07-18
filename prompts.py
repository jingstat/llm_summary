TOPIC_PROMPT_MAP_TEMPLATE = """
You are a helpful assistant that helps retrieve topics talked about in a earnings transcript
- Your goal is to extract the topic names and brief 1-sentence description of the topic
- Topics include:
  - Revenue Performance
  - Profitability
  - Consumer spending 
  - Operating Expenses
  - Market Performance and Competition
  - Product or Service Updates
  - Guidance for Future Quarters
  - Strategic Initiatives
  - Regulatory and Legal Issues
  - Capital Expenditures and Investments
- Provide a brief description of the topics after the topic name. Example: 'Topic: Brief Description'
- Use the same words and terminology that is said in the earnings transcript
- Do not respond with anything outside of the podcast. If you don't see any topics, say, 'No Topics'
- Do not respond with numbers, just bullet points
- Only pull topics from the transcript. Do not use the examples
- Make your titles descriptive but concise. Example: 'Shaan's Experience at Twitch' should be 'Shaan's Interesting Projects At Twitch'
- A topic should be substantial, more than just a one-off comment

% START OF EXAMPLES
 - Consumer spending: Consumers are being cautious with their spending due to uncertain economic environment and inflationary pressures, leading to moderated spending on discretionary categories and shifts to lower-priced items.
% END OF EXAMPLES
"""
TOPIC_PROMPT_COMBINE_TEMPLATE = """
You are a helpful assistant that helps retrieve topics talked about in a earnings transcript
- You will be given a series of bullet topics of topics vound
- Your goal is to exract the topic names and brief 1-sentence description of the topic
- Deduplicate any bullet points you see
- Only pull topics from the transcript. Do not use the examples

% START OF EXAMPLES
 - Consumer spending: Consumers are being cautious with their spending due to uncertain economic environment and inflationary pressures, leading to moderated spending on discretionary categories and shifts to lower-priced items.
% END OF EXAMPLES
"""

TOPIC_SCHEMA = {
    "properties": {
        # The title of the topic
        "topic_name": {
            "type": "string",
            "description" : "The title of the topic listed"
        },
        # The description
        "description": {
            "type": "string",
            "description" : "The description of the topic listed"
        },
        "tag": {
            "type": "string",
            "description" : "The type of content being described",
            "enum" : ['Revenue Related', 'Consumer Related', 'Market Related', 'Regulation', 'Strategy']
        }
    },
    "required": ["topic", "description"],
}

SUMMARY_TEMPLATE = """
You will be given text from a earning transcript which contains many topics.
You goal is to write a summary (5 sentences or less) about a topic the user chooses
Do not respond with information that isn't relevant to the topic that the user gives you
----------------
{context}"""

PRED_PROMPT_CONSUEMR_SPENDING = """
Please determine if the following text is related to consumer spending such as consumer confidence, affordability but do not count company sales.  
"{text}"
Output format: 

Yes or No. 
"""

PRED_PROMPT_LABOR_COST = """
Please determine if the following text is related to labor cost.  
"{text}"
Output format: 

Yes or No. 
"""
