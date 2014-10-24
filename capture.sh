#!/bin/bash

if [ "$1" == "" ] || [ "$2" != "" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

FILE=$1
PREFIX=${FILE%%.*}
SUFFIX=${FILE##*.}

FILE_WITH_NUMBER=${PREFIX}-1
if [ -n "${SUFFIX}" ]; then
  FILE_WITH_NUMBER="${FILE_WITH_NUMBER}.${SUFFIX}"
fi


IMAGE_CAPTURED=0

while [ "${IMAGE_CAPTURED}" == "0" ]; do
  \rm -f ${FILE}
  \rm -f ${FILE_WITH_NUMBER}
  OUTPUT=$(guvcview --no_display -m 1 --exit_on_close -c 1 -i ${FILE} -g /dev/zero -s 1600x1200 -f yuyv 2>&1 > /dev/null)

  if [ -z "$(echo "${OUTPUT}" | grep "Could not grab image")" ]; then
    IMAGE_CAPTURED=1
  else
    echo "Failed to capture an image from the webcam. Retrying..."
  fi
done

mv ${FILE_WITH_NUMBER} ${FILE}

