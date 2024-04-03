from chatbot import chatbot 

# "gpt-3.5-turbo-16k"

def agent(messege,hisory):
  bot=chatbot("",0,"gpt-4")
  response=bot.run(input=messege)
  return response