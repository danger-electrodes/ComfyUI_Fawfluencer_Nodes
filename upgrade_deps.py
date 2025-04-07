import sys
import subprocess

#we need to upgrade protobuf otherwise insightface take its holy time to load
def upgrade_protobuf():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "protobuf==4.25.3"])

#we need to upgrade protobuf otherwise insightface take its holy time to load
def upgrade_mediapipe():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "mediapipe==0.10.21"])

def deps_upgrade():
    upgrade_mediapipe()
    upgrade_protobuf()