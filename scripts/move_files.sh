#!/bin/bash

REMOTE_USER="shanson"
REMOTE_HOST="cuillin"
REMOTE_DIR="/home/shanson/foggie-dynamics/"
LOCAL_DIR="$(dirname "$(dirname "$0")")"  # repo root (one level up from scripts/)

FOLDERS=("results" "logs" "plots")

for FOLDER in "${FOLDERS[@]}"; do
    echo "Syncing ${FOLDER}..."
    rsync -avz --progress \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}${FOLDER}/" \
        "${LOCAL_DIR}/${FOLDER}/"
done
