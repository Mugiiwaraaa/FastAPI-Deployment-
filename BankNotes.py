from pydantic import BaseModel

#defining the class that describes the Bank Note measurements 

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

    