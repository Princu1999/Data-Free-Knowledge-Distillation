
class CSVLogger:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'w') as f:
            f.write("epoch,student_loss,generator_loss,time\n")
    def log(self, epoch, student_loss, generator_loss, time_taken):
        with open(self.path, 'a') as f:
            f.write(f"{epoch},{student_loss},{generator_loss},{time_taken}\n")
