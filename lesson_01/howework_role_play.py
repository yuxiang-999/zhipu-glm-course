"""
实现 role-play 对话数据生成工具：基于用户给定的一段文本，通过调用 ChatGLM 生成两个角色人设，然后调用 CharacterGLM 交替生成这两个角色的对话，并将生成的对话数据保存到当前目录中。

依赖：
pyjwt
requests
zhipuai
python-dotenv

运行方式：
```bash
python howework_role_play.py
```
"""
import itertools
import json
import os
import random
from datetime import datetime

import requests
import jwt
import time
from typing import Literal, TypedDict, List, Generator, Iterator

# 智谱开放平台API key，参考 https://open.bigmodel.cn/usercenter/apikeys
API_KEY: str = os.getenv("API_KEY", "")

"""
相关数据类型的定义
"""


class BaseMsg(TypedDict):
    pass


class TextMsg(BaseMsg):
    """文本消息"""

    # 在类属性标注的下一行用三引号注释，vscode中
    role: Literal["user", "assistant"]
    """消息来源"""
    content: str
    """消息内容"""


TextMsgList = List[TextMsg]


class CharacterMeta(TypedDict):
    """角色扮演设定，它是CharacterGLM API所需的参数"""
    user_info: str
    """用户人设"""
    bot_info: str
    """角色人设"""
    bot_name: str
    """bot扮演的角色的名字"""
    user_name: str
    """用户的名字"""


class RolePlayer:
    name: str
    """角色名"""
    description: str
    """角色人设描述"""
    prologue: str
    """角色开场白"""

    def __init__(self, name, description, prologue):
        self.name = name
        self.description = description
        self.prologue = prologue

    def __str__(self):
        return f"姓名：{self.name}  描述：{self.description}  开场白：{self.prologue}"


class RoleMsg:
    index: int
    """信息序号"""
    name: str
    """角色名"""
    message: str
    """角色回答信息"""

    def __init__(self, index, name, message):
        self.index = index
        self.name = name
        self.message = message

    def __str__(self):
        return f"序号：{self.index}  角色：{self.name}  回答：{self.message}"


RoleMsgList = List[RoleMsg]


"""
调用 chatglm 的方法
"""


class ApiKeyNotSet(ValueError):
    pass


def verify_api_key_not_empty():
    if not API_KEY:
        raise ApiKeyNotSet


def generate_token(apikey: str, exp_seconds: int) -> str:
    # reference: https://open.bigmodel.cn/dev/api#nosdk
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def get_chatglm_response_via_sdk(messages: TextMsgList) -> Generator[str, None, None]:
    """ 通过sdk调用chatglm """
    # reference: https://open.bigmodel.cn/dev/api#glm-3-turbo  `GLM-3-Turbo`相关内容
    # 需要安装新版zhipuai
    from zhipuai import ZhipuAI
    verify_api_key_not_empty()
    client = ZhipuAI(api_key=API_KEY) # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-3-turbo",  # 填写需要调用的模型名称
        messages=messages,
        stream=True,
    )
    for chunk in response:
        yield chunk.choices[0].delta.content


def get_characterglm_response(messages: TextMsgList, meta: CharacterMeta) -> Generator[str, None, None]:
    """ 通过http调用characterglm """
    # Reference: https://open.bigmodel.cn/dev/api#characterglm
    verify_api_key_not_empty()
    url = "https://open.bigmodel.cn/api/paas/v3/model-api/charglm-3/sse-invoke"
    resp = requests.post(
        url,
        headers={"Authorization": generate_token(API_KEY, 1800)},
        json=dict(
            model="charglm-3",
            meta=meta,
            prompt=messages,
            incremental=True)
    )
    resp.raise_for_status()

    # 解析响应（非官方实现）
    sep = b':'
    last_event = None
    for line in resp.iter_lines():
        if not line or line.startswith(sep):
            continue
        field, value = line.split(sep, maxsplit=1)
        if field == b'event':
            last_event = value
        elif field == b'data' and last_event == b'add':
            yield value.decode()


def generate_role_list_prompt(input_text: str) -> str:
    # 基于给定的文段，生成提取人物角色的提示词
    instruction = f"""
在下面文本中，描述了一段故事或场景，其中包含两个以上的人物，请推断并生成各人物的“角色人设”和“开场白”。要求：
1. 根据故事或场景，推断出每个人物的“角色人设”和“开场白”。
2. “角色人设”要包含性别、年龄、性格特点、外貌特征、爱好和特长等，尽量用短语描写，而不是完整的句子。
3. “角色人设”和“开场白”不能包含敏感词，并且不能超过50字。
4. 至少要生成两个“角色人设”。
5. 返回结果要符合json格式。

文本：
{input_text}

生成角色人设列表 (json格式)：
{{
  "characters": [
    {{
      "name": "角色1的姓名",
      "description": "角色1的人设描述。",
      "prologue": "角色1的开场白。"
    }},
    {{
      "name": "角色2的姓名",
      "description": "角色2的人设描述。",
      "prologue": "角色2的开场白。"
    }},
    ...
  ]
}}
"""
    return instruction.strip()


def generate_role_list(role_list_prompt: str) -> Generator[str, None, None]:
    """ 用 chatglm 生成角色人设 """

    return get_chatglm_response_via_sdk(
        messages=[
            {
                "role": "user",
                "content": role_list_prompt.strip()
            }
        ]
    )


def generate_character_meta(bot_name: str, bot_info: str, user_name: str, user_info: str) -> CharacterMeta:
    character_meta = {
        "bot_name": f"{bot_name}",
        "bot_info": f"{bot_info}",
        "user_name": f"{user_name}",
        "user_info": f"{user_info}"
    }
    return character_meta


def switch_message_role(messages: TextMsgList):
    for msg in messages:
        if msg["role"] == "user":
            msg["role"] = "assistant"
        else:
            msg["role"] = "user"


def output_stream_response(response_stream: Iterator[str]):
    response = ""
    for content in itertools.accumulate(response_stream):
        response = content
    return response


def role_chat(input_text: str, chat_rounds: int):

    # 基于给定的文段，生成提取人物角色的提示词
    role_list_prompt = generate_role_list_prompt(input_text)
    if not role_list_prompt:
        print("调用 chatglm 生成提取人物角色的提示词出错")
        return

    print(f'提取角色人设的提示词：\n{role_list_prompt}')

    # 根据文段内容，生成角色人设列表
    role_list_json = "".join(generate_role_list(role_list_prompt))
    print(f'chatglm返回的结果：\n{role_list_json}')

    # 解析JSON字符串
    role_list_data = json.loads(role_list_json.strip())

    # 创建RolePlayer对象列表
    role_players = [RolePlayer(character["name"], character["description"], character["prologue"])
                    for character in role_list_data["characters"]]

    # 打印RolePlayer对象列表
    for player in role_players:
        print(player)

    # 从role_players列表中随机选择两个RolePlayer对象
    selected_players = random.sample(role_players, 2)
    for index, selected_player in enumerate(selected_players, start=1):
        print(f'挑选的角色{index}为 -> {selected_player}')

    # 对话历史记录
    messages = []
    session_messages: RoleMsgList = []

    role_player_1: RolePlayer = selected_players[0]
    role_player_2: RolePlayer = selected_players[1]

    for index_round in range(chat_rounds):
        switch_message_role(messages)
        character_meta = None

        if index_round % 2 == 1:
            # 角色1为机器人
            character_meta = generate_character_meta(
                bot_name=role_player_1.name,
                bot_info=role_player_1.description,
                user_name=role_player_2.name,
                user_info=role_player_2.description
            )
        else:
            # 角色2为机器人
            character_meta = generate_character_meta(
                bot_name=role_player_2.name,
                bot_info=role_player_2.description,
                user_name=role_player_1.name,
                user_info=role_player_1.description
            )

            if index_round == 0:
                # 首轮对话
                messages.append(TextMsg({"role": "user", "content": role_player_1.prologue}))
                session_messages.append(RoleMsg(index_round, role_player_1.name, role_player_1.prologue))

        response_stream = get_characterglm_response(messages=messages, meta=character_meta)
        bot_response = output_stream_response(response_stream)
        messages.append(TextMsg({"role": "assistant", "content": bot_response}))
        session_messages.append(RoleMsg(index_round + 1, character_meta['bot_name'], bot_response))

    # 打印对话记录
    print("对话记录：")
    for session_message in session_messages:
        print(session_message)

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 构造带有时间信息的文件名
    file_name = f"session_messages_{current_time}.json"

    # 获取当前目录路径
    current_directory = os.getcwd()

    # 构造完整的文件路径
    file_path = os.path.join(current_directory, file_name)

    # 将角色对话记录转为JSON格式的字符串
    session_messages_json = json.dumps([vars(session_message) for session_message in session_messages])

    # 将JSON字符串写入文件
    with open(file_path, "w") as file:
        file.write(session_messages_json)

    # 打印保存的文件路径
    print("角色对话记录已保存到:", file_path)


if __name__ == "__main__":
    # 文本场景
    input_text: str = """
到了夏天美好的黄昏时刻，闷热的街头巷尾都空荡荡的，只有女佣在大门口踢毽子。他打开窗户，凭窗眺望，看见底下的小河流过桥梁栅栏，颜色有黄有紫有蓝，
使鲁昂这个街区变成了见不得人的小威尼斯。有几个工人蹲在河边洗胳膊。阁楼里伸出去的竿子上，晾着一束一束的棉线。对面屋顶上是一望无际的青天，
还有一轮西沉的红日。乡下该多好呵！山毛榉下该多凉爽呵！他张开鼻孔去吸田野的清香，可惜只闻到一股热气。
"""
    # 对话轮数
    chat_rounds: int = 10

    if API_KEY:
        role_chat(input_text, chat_rounds)
    else:
        print("未设置API_KEY")
