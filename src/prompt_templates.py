from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def _get_structured_prompt(query: str, parser: PydanticOutputParser) -> str:
    prompt_template = PromptTemplate(
            template=(
                f"{query}\n\n"\
                f"Поверни результат у форматі JSON згідно з цією схемою:\n"\
                "{schema}"
            ),
            input_variables=[],
            partial_variables={"schema": parser.get_format_instructions()}
        )
    return prompt_template.format()
 
def get_text_template(parser: PydanticOutputParser = None) -> str:

    query = "Витягни текст документу в порядку його прочитання."
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

def get_table_template(parser: PydanticOutputParser = None) -> str:

    query = "Витягни таблицю з даними в форматі HTML."
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

def get_title_template(parser: PydanticOutputParser = None) -> str:

    query = "Витягни **назви** для усіх розділів документу, таблиці і графіку **в порядку прочитання**."
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

def get_class_template(parser: PydanticOutputParser = None) -> str:

    query = "Віднеси текст зображення до одного із наступних класів: \n"\
            "Політика; Програмування; Фінанси; Література; Спорт; Археологія України; Західне мистецтво; Природні ресурси; Космос; Економіка.\n"\
            "Відповідай лише назвою класу."
    
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

class TextSchema(BaseModel):
    text: str = Field(description="Tекст документу.")

class TableSchema(BaseModel):
    table: str = Field(description="Таблиця з даними в форматі HTML.")

class TitleSchema(BaseModel):
    titles: list = Field(description="Назви для усіх розділів документу, таблиці і графіку в порядку прочитання.")

class ClassSchema(BaseModel):
    classes: list = Field(description="Назва класу для тексту зображення.")


if __name__ == "__main__":
    parser = PydanticOutputParser(pydantic_object=ClassSchema)
    prompt = get_class_template(parser)
    print(prompt)

    prompt = get_class_template()
    print(prompt)