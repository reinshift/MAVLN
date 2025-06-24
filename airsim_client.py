"""
when env_airsim simulator is started, this script will play a role as Client, 
get images from the Server, and feed them to the model with instructions to take actions.
finally, send the actions to the Server.
"""
import airsim
import logging
import socket
import json
import base64
import io
import time
import numpy as np
from PIL import Image
import sys
import os
import threading

logger = logging.getLogger(__name__)

