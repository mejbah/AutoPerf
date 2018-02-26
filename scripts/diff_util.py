import sys
import os
import subprocess
import re

def startFunc(original, updated):
  
  files = []
  file_types = [ '*.c', '*.cpp', '*.hh', '*.h' ]    
  find_command = "find"
  find_expression = "-name"
  find_location = original

  ## files in original
  for type in file_types:
    processFind = subprocess.Popen([find_command, find_location, find_expression, type], stdout = subprocess.PIPE)
    stdout, stderr = processFind.communicate()
    filenames = stdout.split('\n')
    for filename in filenames:
      if len(filename) > 0:
        files.append( filename )

  ## diff with updated
  diffBlocks = {} #key-filename : val-[list of fucntions]
  for file in files:
    diffOut = doDiff(original, updated, file)
    if len(diffOut)!=0:
      funcNames = getDiffFuncNames(diffOut)
      if len(funcNames) > 0:
        diffBlocks[file] = funcNames  

  funcDict = {} 
  for key in diffBlocks:
    for funcName in diffBlocks[key]:
      ##print key,  funcName         ##key = filename
      funcDict[funcName] = key

  ##print functions changed + corresponding filename
  for key in funcDict:
    print key, funcDict[key]

  ##print files changed
  #for key in diffBlocks:
  #  print key

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def getDiffFuncNames(diffOut):
  funcDict = {}
  ##compile regex for funciton
  r = re.compile("@@.*.@@.*.\(.*.\).*")
  for line in diffOut.split('\n'):
    if r.match(line):
      funcName = line.split('@@')[-1]
      if funcName in funcDict:
        funcDict[funcName].append(line)
      else: 
        funcDict[funcName] = []
        funcDict[funcName].append(line)
  return funcDict


def doDiff(original, updated, file):

  diff_command = "diff"
  diff_options = "-NrpbB"
  opt_context = "-U0"
  opt_ignore = "-I"
  opt_ignore_comment = "* "
  opt_ignore_line_comment = "\/\/"
  #trim original
  trimmedFileName = file[len(original):]
  originalFileName = file
  updatedFileName = updated + trimmedFileName

  #print diff_command, diff_options, opt_context, opt_ignore, opt_ignore_comment, opt_ignore, opt_ignore_line_comment, originalFileName, updatedFileName
  processDiff = subprocess.Popen([ diff_command, diff_options, opt_context, opt_ignore, opt_ignore_comment, opt_ignore, opt_ignore_line_comment, originalFileName, updatedFileName ], stdout = subprocess.PIPE)

  stdout,stderr = processDiff.communicate()
  if stderr != None:
     print diff_command, diff_options, opt_context, opt_ignore, opt_ignore_comment, opt_ignore, opt_ignore_line_comment, originalFileName, updatedFileName
     print stderr
     sys.exit(0)
    
  return stdout

if __name__ == "__main__":
  
  original = sys.argv[1]
  updated = sys.argv[2]

  startFunc(original, updated)
