
import smtplib

#Sends and email to you, to let you know when a task is finished executing, I used gmail for this
def send_mail(script,method,dataset,error = False):

  sender_add='jacquelineruiqi@gmail.com'
  receiver_add='jacquelineruiqi@gmail.com' 
  password='1100888168' #you'll have to genearte your own password for gmail and replace the above addresses with your own

  smtp_server=smtplib.SMTP("smtp.gmail.com",587)
  smtp_server.ehlo() #setting the ESMTP protocol
  smtp_server.starttls() #setting up to TLS connection
  smtp_server.ehlo() #calling the ehlo() again as encryption happens on calling startttls()
  smtp_server.login(sender_add,password) #logging into out email id
  msg_to_be_sent = "Subject: {} Has Finished Testing {} on the {} dataset".format(script,method,dataset)
  if error:
    msg_to_be_sent += ' ERROR'


  #sending the mail by specifying the from and to address and the message 
  smtp_server.sendmail(sender_add,receiver_add,msg_to_be_sent)
  smtp_server.quit()#terminating the server


