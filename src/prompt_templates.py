from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def _get_structured_prompt(query: str, parser: PydanticOutputParser) -> str:

    template = """
    {query} 
    Поверни результат у форматі JSON згідно з цією схемою:
    {schema}
    """.strip()

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"schema": parser.get_format_instructions()}
    )
    return prompt_template.format(query=query)
 
def get_text_template(parser: PydanticOutputParser = None) -> str:

    query = """
        Витягни текст зображення у точному вигляді.
        - Залиши слова, літери й розділові знаки такими, як вони є.
        - Збережи порядок, великі й малі літери, пробіли, розриви рядків.
        - Будь-яке редагування, переклад, дописування або зміна заборонені.
        - Якщо текст містить таблицю, то форматуй її у Markdown.

        Поверни лише текст дослівно.
    """
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

def get_table_template(parser: PydanticOutputParser = None) -> str:

    query = """
        Витягни таблицю з даними, що міститься на зображенні, у форматі HTML.

        Вимоги:
        - Поверни лише HTML-код таблиці, без додаткового тексту чи опису.
        - Залиш текст у комірках таким, як він є на зображенні.
        - Збережи структуру таблиці, включно з усіма рядками (`<tr>`) та комірками (`<td>`), відповідно до оригінального вигляду.
        - Уникай використання тегів для заголовків (`<thead>`, `<th>`).
        - Використовуй тег `<table>` для обгортки всієї таблиці.
        - Комірки без вмісту відображай як `<td></td>`.
        - Залиш багаторядкові комірки у вихідному вигляді, зберігаючи розриви рядків.
        
        Важливо:
        Будь-яке редагування, переклад, форматування чи пояснення недопустимі. Тільки HTML-таблиця.
    """
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

def get_title_template(parser: PydanticOutputParser = None) -> str:

    query = """
        Витягни назви всіх розділів документу, таблиць і графіків у порядку, в якому вони з’являються в документі.

        Вимоги:
        - Витягни лише ті елементи, які виглядають як заголовки, підписи або назви секцій документу.
        - Залиш назви у тому вигляді, в якому вони подані у документі.
        - Збережи точний порядок прочитання — зверху вниз, зліва направо.
        - Уникай інтерпретацій, описів чи узагальнень.
        - Поверни тільки список назв розділів без додаткового тексту.
        - Кожна назва має починатись з нового рядка.

        Важливо:
        Текст кожної назви має залишатися без змін. Будь-які правки або перекручення заборонено.
    """
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

def get_class_template(parser: PydanticOutputParser = None) -> str:

    query = """
        Проаналізуй текст на зображенні та віднеси його до одного з наведених класів:
        Політика; Програмування; Фінанси; Література; Спорт; Археологія України; Західне мистецтво; Природні ресурси; Космос; Економіка.

        Вимоги:
        - Обери лише один клас із наведеного списку.
        - Поверни тільки назву класу без коментарів, пояснень або додаткового тексту.
        - Назва класу має збігатися точно з формулюванням у списку.

        Важливо:
        Будь-яке форматування, пояснення чи переформулювання неприйнятні. У відповіді має бути лише чиста назва одного класу.
    """
    
    if parser is None:
        return query
    
    return _get_structured_prompt(query, parser)

class TextSchema(BaseModel):
    text: str = Field(description="Tекст документу.")

class TableSchema(BaseModel):
    table: str = Field(description="Таблиця з даними в форматі HTML.")

class TitleSchema(BaseModel):
    titles: str = Field(description="Назви для усіх розділів документу, таблиці і графіку в порядку прочитання.")

class ClassSchema(BaseModel):
    classes: str = Field(description="Назва класу для тексту зображення.")


if __name__ == "__main__":
    parser = PydanticOutputParser(pydantic_object=ClassSchema)
    prompt = get_class_template(parser)
    print(prompt)

    prompt = get_class_template()
    print(prompt)