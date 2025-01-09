import argparse
import atexit
import hashlib
import hmac
import json
import os

import ngrok  # Assuming there's a similar way to use ngrok with Flask

import requests  # We'll use requests instead of httpx for HTTP calls
from flask import Flask, request, jsonify, abort

from datetime import datetime
import main as DTC
import util.smartparse as smartparse
def default_params():
    params=smartparse.obj()
    params.cdr_callback_port=9999
    params.cdr_callback_system_name="DTC_APP"
    params.cdr_callback_registration_secret="mysecret_dtc_app"
    params.cdr_callback_api_token=""
    params.ngrok_token=""
    params.cdr_callback_url=""
    return params


params = smartparse.merge(DTC.params, default_params())



class Settings:
    system_name: str = params.cdr_callback_system_name
    system_version: str = "2.1.0_%s"%(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    local_port: int = params.cdr_callback_port
    callback_url: str = params.cdr_callback_url + "/hook"
    registration_secret: str = '%s_%s'%(params.cdr_callback_registration_secret,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    user_api_token: str = params.cdr_callback_api_token
    cdr_host: str = params.cdr_endpoint
    registration_id: str = ""

app_settings = Settings()

# Setting up ngrok URL
if not params.ngrok_token=="":
    listener = ngrok.forward(app_settings.local_port, authtoken=params.ngrok_token)
    app_settings.callback_url = listener.url() + "/hook"

print(app_settings.callback_url)

def clean_up():
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    requests.delete(f"{app_settings.cdr_host}/user/me/register/{app_settings.registration_id}", headers=headers)

atexit.register(clean_up)

app = Flask(__name__)

def event_handler(evt):
    try:
        if evt.get("event") == "ping":
            print("Received PING!")
        elif evt.get("event") == "document.process":
            print("Received document process event!")
            print(evt.get("payload", {}))
            DTC.app.process(evt.get("payload", {})['id'])
        else:
            print("Nothing to do for event: %s", evt)
    except Exception as e:
        print(f"background processing event: {evt}, exception: {e}")
        raise

def verify_signature(request):
    signature_header = request.headers.get("x-cdr-signature-256")
    payload_body = request.data

    if not signature_header:
        abort(403, description="x-hub-signature-256 header is missing!")

    hash_object = hmac.new(app_settings.registration_secret.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = hash_object.hexdigest()

    if not hmac.compare_digest(expected_signature, signature_header):
        abort(403, description="Request signatures didn't match!")

@app.route("/hook", methods=["POST"])
def hook():
    verify_signature(request)
    evt = request.json
    # Background task could be replaced with a proper async task like Celery if needed
    event_handler(evt)
    return jsonify(ok="success")

def run():
    app.run(host="0.0.0.0", port=app_settings.local_port, debug=False)

def register_system():
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    registration = {
        "name": app_settings.system_name,
        "version": app_settings.system_version,
        "callback_url": app_settings.callback_url,
        "webhook_secret": app_settings.registration_secret,
        "auth_header": "",
        "auth_token": "",
        "events": []
    }

    response = requests.post(f"{app_settings.cdr_host}/user/me/register", json=registration, headers=headers)
    app_settings.registration_id = response.json().get("id")
    assert not app_settings.registration_id is None
    print("APP ID %s"%app_settings.registration_id)

if __name__ == "__main__":
    
    register_system()
    run()

