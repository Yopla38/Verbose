import json
import paramiko

class SSHClient:
    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password

    def generate_text(self, prompt):
        client = paramiko.Transport(self.host)
        client.connect(username=self.username, password=self.password)
        channel = client.open_channel(kind="session")
        channel.send(json.dumps({"prompt": prompt}))
        response = channel.recv(1024).decode("utf-8")
        client.close()
        return json.loads(response)["response"]


client = SSHClient("192.168.1.100", "username", "password")
response = client.generate_text("Bonjour, comment vas-tu ?")
print(response)