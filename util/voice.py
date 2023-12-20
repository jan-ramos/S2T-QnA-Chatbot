from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd


class Record():
        def __init__(self,sampling_rate):
                self.freq = sampling_rate
                self.duration = 5
                self.record()
                
        def record(self):
                print('Recording')
                self.recording = sd.rec(int(self.duration * self.freq), 
                                        samplerate=self.freq, channels=1)
                sd.wait()
                print('Recording Over')

        def output(self):
                return self.recording
                
class Whisper():
        def __init__(self):
                print('Loading Model')
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                self.model.config.forced_decoder_ids = None
                print('Model Loaded')

        def process(self,speech,sampling_rate):
                print('Processing Voice')
                input_features = self.processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_features 
                predicted_ids = self.model.generate(input_features)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                print('Process Complete')
                print('Output')
                return transcription