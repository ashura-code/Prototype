from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

import re

def normalize_query(text: str) -> str:
    # Lowercase and remove numbers (and optionally stopwords, etc.)
    text = text.lower()
    text = re.sub(r'\b\d+\b', 'N', text)  # Replace digits with 'N'
    return text

log_examples = [
    "What are the anomalies in the month of April?",
    "Show me unusual patterns in the logs.",
    "Are there any spikes in traffic this week?",
    "Which requests failed the most?",
    "Any suspicious behavior detected today?",
    "What are the top endpoints being accessed?",
    "Are there abnormal request patterns?",
    "Show me all failed login attempts.",
    "Which users accessed the system the most?",
    "Any increase in rejected packets?",
    "Which IPs had the most outbound traffic?",
    "How many VPC flow logs show denied traffic?",
    "What source IPs were blocked in the last 24 hours?",
    "Top destinations from VPC logs?",
    "Show me all connections that were accepted.",
    "What ports were most frequently targeted?",
    "Analyze inbound vs outbound traffic volume.",
    "Show rejected connections by region.",
    "Get me VPC logs from last Friday.",
    "Who accessed the private subnet?",
    "How many executions failed yesterday?",
    "What’s the average response duration per endpoint?",
    "Show logs where status code is 500.",
    "Which functions are taking too long to run?",
    "Find all executions that ended in errors.",
    "Which API calls succeeded with 200 status?",
    "List all functions invoked by user ID 12345.",
    "Get the last 10 error logs from execution.",
    "Who triggered the most failed functions?",
    "What’s the request volume trend this month?"
]
# chart_query_examples = [
#     "Show the trend of failed login attempts per day over the last month",
#     "Visualize the number of requests per endpoint every hour",
#     "Plot the number of requests rejected per day in the last week",
#     "Chart the distribution of request durations per method type",
#     "Show the volume of traffic per source IP every 30 minutes",
#     "Trend of function execution failures per endpoint",
#     "Plot the daily number of requests per user over the last month",
#     "Visualize the number of successful requests per region",
#     "Chart the distribution of HTTP status codes by hour",
#     "Visualize the average duration of requests per user over time",
#     "Show the trend of accepted vs rejected requests over the week",
#     "Plot the number of requests per service endpoint over the last 7 days",
#     "Show the variation in request sizes over the past month",
#     "Chart the number of requests grouped by source IP and hour",
#     "Plot failed authentication attempts per day over the last 2 weeks",
#     "Visualize the number of connections accepted vs rejected per day",
#     "Show the time series of VPC action counts per day",
#     "Chart the request volume per method type (GET, POST, etc.) every hour",
#     "Plot the correlation between request duration and response status",
#     "Visualize the frequency of requests from each country in the last 7 days",
#     "Trend of latency spikes across different service endpoints",
#     "Chart hourly traffic distribution for the past month",
#     "Visualize the number of dropped packets per destination port every 10 minutes",
#     "Show the trend of rejected requests per source IP",
#     "Plot the number of access requests per method type across different days",
#     "Visualize the number of failed requests by endpoint per day",
#     "Trend of the number of requests per user over time",
#     "Show hourly breakdown of traffic volume per IP",
#     "Plot spikes in VPC rejects and accepts on a weekly basis",
#     "Visualize the average latency per service endpoint per hour",
#     "Chart requests to blocked ports per day over the last week",
#     "Show the number of function calls per user over time",
#     "Plot the total traffic by country across different days"
# ]

log_embeddings = model.encode(log_examples, convert_to_tensor=True)
# chart_embeddings = model.encode(chart_query_examples, convert_to_tensor=True)

def is_relevant_log_query(question: str) -> bool:
    normalized_query = normalize_query(question)
    query_embedding = model.encode(normalized_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, log_embeddings)
    max_score = scores.max().item()
    return max_score > 0.5  # You can adjust this threshold

print(is_relevant_log_query("list ids from the log"))
# def is_relevant_chart_query(chart_query)->bool:
#     query_embedding = model.encode(chart_query, convert_to_tensor=True)
#     scores = util.cos_sim(query_embedding, chart_embeddings)
#     max_score = scores.max().item()
#     return max_score > 0.5  # Threshold can be tuned




# chart_query_examples = [
#     "Show the number of rejected requests per day for the last month.",
#     "Plot the trend of failed login attempts over the past week.",
#     "Visualize the top 10 endpoints by number of hits.",
#     "Compare accepted vs rejected VPC requests by protocol.",
#     "Show failed function executions grouped by status code.",
#     "Plot the count of requests by region for today.",
#     "Visualize the number of bytes transferred per hour.",
#     "Compare execution durations for different functions.",
#     "Show a pie chart of traffic distribution by destination port.",
#     "Display a bar chart of request count per user ID.",
    
#     "Plot number of failed API calls per endpoint over the last 7 days.",
#     "Show a time series of anomaly scores over the past month.",
#     "Visualize the frequency of status codes in access logs.",
#     "Display top 5 IP addresses by denied access attempts.",
#     "Show the distribution of source IPs hitting the service.",
#     "Visualize count of successful vs failed function calls per day.",
#     "Compare response time averages by HTTP method.",
#     "Plot traffic volume trend per availability zone.",
#     "Group and visualize failed VPC connections by source IP.",
#     "Chart execution error types and their frequencies.",
    
#     "Display request volume over time grouped by endpoint.",
#     "Plot correlation between execution time and failure rate.",
#     "Compare user activity patterns by time of day.",
#     "Visualize number of VPC log entries by action type (ACCEPT/REJECT).",
#     "Show a histogram of function execution durations.",
#     "Compare rejected packets by source and destination port.",
#     "Plot request distribution across different timezones.",
#     "Visualize traffic anomalies using a time series chart.",
#     "Compare error logs per module over the last 30 days.",
#     "Plot success vs failure trends over time for API calls.",
#     "List the number of access attempts by each user in April.",
#     "What are the most frequently accessed endpoints this week?",
#     "List the IPs with the highest number of failed requests.",
#     "What are the peak hours for incoming traffic?",
#     "List the days with the most failed function calls.",
#     "What are the most common reasons for access rejections?",
#     "List the regions with the most rejected VPC connections.",
#     "What are the average response times per endpoint?",
#     "Give me the top source IPs grouped by number of hits.",
#     "List the functions with the longest average execution time.",

#     "What are the most frequent status codes returned in March?",
#     "List the endpoints with the highest error rate.",
#     "What are the top 10 users by total traffic generated?",
#     "List the number of successful logins per user this week.",
#     "What are the most common destination ports for failed access?",
#     "Give me a count of actions by type for each day.",
#     "List the anomaly events detected per hour.",
#     "What are the execution durations grouped by function name?",
#     "List the top 5 endpoints causing execution failures.",
#     "What are the usage trends across different time windows?",

#     "Give me the distribution of response sizes over the past 7 days.",
#     "List the number of VPC rejections grouped by protocol.",
#     "What are the sources with the most unauthorized access attempts?",
#     "List all request methods used and their frequencies.",
#     "What are the endpoints with the most consistent failures?",
#     "Give me the hourly traffic volume for the last 24 hours.",
#     "What are the trends in failed login attempts per region?",
#     "List the action types seen in logs and their counts.",
#     "What are the IPs with unusual traffic patterns?",
#     "Give me failed executions grouped by day and function."
# ]













