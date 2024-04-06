from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import json
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse

import wikipedia

import sys

from utils.processing import tokenize_text
import trafilatura
from summary_app.inference import SummaryExtractor
import json


def index(request):
    return HttpResponse("Hello, world. You're at the wiki index.")


# https://pypi.org/project/wikipedia/#description
def get_wiki_summary(request):
    topic = request.GET.get('topic', None)

    print('topic:', topic)

    data = {
        'summary': tokenize_text(wikipedia.summary(topic, sentences=1)),
        'raw': 'Successful',
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)


def set_config(request):
    pass


def summarize_page(request):
    url_str = request.GET.get("url", None)
    if url_str is not None:

        # URL of the webpage you want to parse
        # url = "https://e.vnexpress.net/photo/football/8-potential-replacements-for-coach-troussier-at-vietnam-national-team-4728112.html"

        # Download and parse the webpage using Trafilatura
        html_content = trafilatura.fetch_url(url_str)
        parsed_content = trafilatura.extract(html_content)
        result = SummaryExtractor.get_instance().get_result(parsed_content)
        print(result)
        data = {
            "result": result
        }
        return JsonResponse(data)
    return JsonResponse({})

def summarize_content(request):
    req_body = request.body.decode("utf-8")
    req_body = json.loads(req_body)
    content = None
    try:
        content = req_body["content"]
    except:
        print("`content` not found in request body")
    if content is not None:
        result = SummaryExtractor.get_instance().get_result(content)
        print(result)
        data = {
            "result": result
        }
        return JsonResponse(data)
    return JsonResponse({})

