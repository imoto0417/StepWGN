#!/bin/bash
echo "# StepWGN" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/imoto0417/StepWGN.git
git push -u origin master
