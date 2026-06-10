#!/usr/bin/env bash

if [ "$2" == "--xml" ]; then
  cat "$1.xml"
else   
  case "$1" in
    HubFederated)
      uv run hub.py "${@:2}"
      ;;
    ClientFederated)
      uv run client.py "${@:2}"
      ;;
    *)
      cat slicer_cli_list.json
      ;;
  esac      
fi  
