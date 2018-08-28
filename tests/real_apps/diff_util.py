import sys
import os
import subprocess
import re
from sets import Set

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
  diffBlocks = {} ## key-filename : val-list of funcnames changed
  for file in files:
    diffOut = doDiff(original, updated, file)
    if len(diffOut)!=0:
      funcSet = getFuncDiff(diffOut)
      if len(funcSet) > 0:
        diffBlocks[file] = funcSet

  funcDict = {} 
  for key in diffBlocks:
    for funcName in diffBlocks[key]:
      funcDict[funcName] = key ##TODO: one function name can be in multiple files, here we are storing only one

  ##print functions changed + corresponding filename
  #for key in funcDict:
  #  print key, funcDict[key]

  ##print files changed
  for key in diffBlocks:
      print key, diffBlocks[key]

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def recoverFuncNameFromDiffLine(line):
  ## find the first occurance of '(' while from the end to begining
  lineEndWithFuncName = line.split('(')[-2]
  funcName = lineEndWithFuncName.rstrip().split(' ')[-1] ## function name is followed by a space before starting '(' for arguments, so use rstrip()
  if len(funcName)==0: 
    print "Error?? No function name found in:", line
  if funcName == '=':
    print "Error?? No function name found in:", line
    funcName = "" ## not a function, some assignment somehow shown in diff output as the funcname?? ##TODO: fix it if really needed
  return funcName
  #return line.split('@@')[-1]

def getFuncDiff(diffOut):
  funcSet = Set()
  ##compile regex for funciton
  #r = re.compile("@@.*.@@.*.\(.*.\).*") ## function's end bracket might not present in the fisrt line where the name is present
  r = re.compile("@@.*.@@.*.\(.*")
  for line in diffOut.split('\n'):
    if r.match(line):
      funcName = recoverFuncNameFromDiffLine(line) #line.split('@@')[-1]
      if funcName != "":
        funcSet.add(funcName)
  return funcSet


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

  ## ignore comments
  #print diff_command, diff_options, opt_context, opt_ignore, opt_ignore_comment, opt_ignore, opt_ignore_line_comment, originalFileName, updatedFileName
  processDiff = subprocess.Popen([ diff_command, diff_options, opt_context, opt_ignore, opt_ignore_comment, opt_ignore, opt_ignore_line_comment, originalFileName, updatedFileName ], stdout = subprocess.PIPE)
  ## do not ignore comments
  #processDiff = subprocess.Popen([ diff_command, diff_options, opt_context, originalFileName, updatedFileName ], stdout = subprocess.PIPE)
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
