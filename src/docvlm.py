from models import qwen2vl_unsloth as qww
import docdataset as dd

dataset = dd.download_dataset()['train']


def prompt_qwen(image):

    result = qww.process_doc_image(image, model)

    if result:
        print("result:", result)
        # print("Titles:", result.titles)
        # print("Text Preview:", result.text[:300])
        # print("Table HTML:", result.table[:300])
    else:
        print("Failed to parse the response.")

if __name__ == "__main__":
    dataset = dd.download_dataset()['train']
    image = dataset[-10]["image"].convert("RGB")

    model = qww.QwenVL2_LLM(
        model_name = "unsloth/Qwen2-VL-7B-Instruct",
        max_new_tokens = 2048,
        device = "cuda",
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth")
    
    result = qww.process_doc_image(image, model)

    if result:
        print("result:", result)
        # print("Titles:", result.titles)
        # print("Text Preview:", result.text[:300])
        # print("Table HTML:", result.table[:300])
    else:
        print("Failed to parse the response.")

