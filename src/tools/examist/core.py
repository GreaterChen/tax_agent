from asyncio import get_child_watcher
import time

# 处理相对导入和绝对导入
try:
    # 相对导入（作为模块使用时）
    from .Filter import Filter
    from .Select_Page import Select_Page
    from .Get_Page import Get_Page
    from .Select_Law import Select_Law
    from .Final import Final
    from .Get_Hint import Get_Hint
    from .Get_Law import Get_Law
    from .Get_Law_Indices import Get_Law_Indices
except ImportError:
    # 绝对导入（直接运行时）
    from Filter import Filter
    from Select_Page import Select_Page
    from Get_Page import Get_Page
    from Select_Law import Select_Law
    from Final import Final
    from Get_Hint import Get_Hint
    from Get_Law import Get_Law
    from Get_Law_Indices import Get_Law_Indices

async def core(input:str)-> dict:
    # 开始计时
    start_time = time.time()
    print(f"{'='*60}")
    print(f"Examist1 Start")
    print(f"开始时间: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"{'='*60}")
    
    # 步骤1: Filter
    step_start = time.time()
    print(f"正在执行 Filter ...")
    filter_output=await Filter(input)
    step_end = time.time()
    step_duration = step_end - step_start
    total_duration = step_end - start_time
    
    filter_result=filter_output['result']
    filter_content=filter_output['content']
    filter_fields=filter_output['fields']
    Hint = Get_Hint(filter_fields)
    print(f"Filter 完成 (本步: {step_duration:.2f}秒, 累计: {total_duration:.2f}秒)")
    print(f"      -> filter_result: {filter_result}")
    print(f"      -> filter_content 长度: {len(filter_content)} 字符")
    print(f"{'-'*60}")
    
    if filter_result=='fail':
        print(f"Filter 结果为 fail，直接返回结果")
        Final_content = filter_content
        status = '0'
        final_time = time.time()
        total_duration = final_time - start_time
        print(f"{'='*60}")
        print(f"处理完成！总耗时: {total_duration:.2f}秒")
        print(f"最终状态: {status}")
        print(f"{'='*60}")
    else:
        Question = filter_content
        print(f"[2/6] Filter 成功，提取的问题: {Question[:100]}...")
        print(f"*-*-*-* Hint: {Hint}")

        # 步骤3: Select_Page
        step_start = time.time()
        print(f"[3/6] 正在执行 Select_Page 函数...")
        Select_Page_Result = await Select_Page(Question)
        step_end = time.time()
        step_duration = step_end - step_start
        total_duration = step_end - start_time
        
        Page_List=Select_Page_Result['Request']
        print(f"[3/6] Select_Page 函数完成 (本步: {step_duration:.2f}秒, 累计: {total_duration:.2f}秒)")
        print(f"      -> Page_List: {Page_List}")
        print(f"{'-'*60}")

        # 步骤4: Get_Page
        step_start = time.time()
        print(f"[4/6] 正在执行 Get_Page 函数...")
        Pages = Get_Page(Page_List)
        step_end = time.time()
        step_duration = step_end - step_start
        total_duration = step_end - start_time
        print(f"[4/6] Get_Page 函数完成 (本步: {step_duration:.2f}秒, 累计: {total_duration:.2f}秒)")
        print(f"      -> Pages 长度: {len(Pages)} 字符")
        print(f"{'-'*60}")

  
        
        #Law 处理步骤
        # 步骤5: Select_Law
        step_start = time.time()
        print(f"正在执行 Select_Law ...")
        # 此处还需要插入Law_Indices: Law_Indices = Get_Law_Indices() 
        Law_Indices = Get_Law_Indices()
        Pages = Pages + Hint
        Selected = await Select_Law(Question,Pages,Law_Indices)

        step_end = time.time()
        step_duration = step_end - step_start
        total_duration = step_end - start_time
        
        # --------施工中--------

        Notes = Selected["Notes"]  
        Selected_Laws =  Selected["Laws"]
        Laws = Get_Law (Selected_Laws)


        print(f"[5/6] Select_Law 函数完成 (本步: {step_duration:.2f}秒, 累计: {total_duration:.2f}秒)")
        print(f"      -> Notes 长度: {len(Notes)} 字符")
        print(f"{'-'*60}")

       # Laws = Get_Law(Law_List)
        

        #--------施工中--------
        
        # 步骤6: Final
        step_start = time.time()
        print(f"[6/6] 正在执行 Final 函数...")
        Answer = await Final(Question,Pages,Notes,Laws)
        step_end = time.time()
        step_duration = step_end - step_start
        total_duration = step_end - start_time
        
        Answer_Content = Answer['Final_Answer']
        Answer_Reasoning = Answer['Reasoning']
        print(f"[6/6] Final 函数完成 (本步: {step_duration:.2f}秒, 累计: {total_duration:.2f}秒)")
        print(f"      -> Answer_Content 长度: {len(Answer_Content)} 字符")
        print(f"{'-'*60}")

        Final_content = Answer_Content
        Final_status = '1'
        status=Final_status

        final_time = time.time()
        total_duration = final_time - start_time
        print(f"{'='*60}")
        print(f"处理完成！总耗时: {total_duration:.2f}秒")
        print('******************************')
        print(f"最终状态: {Final_status}")
        print(f"{'='*60}")
    
    Return_Dict = {'status':status,'content':Final_content}
    return Return_Dict








#---------------TEST---------------------------------

if __name__ == "__main__":
    import asyncio
    
    async def test_core():
        # 测试输入文本 - 可以自行编辑
        test_input = '''
Fact:

Betty is a fresh university graduate majoring in visual arts. In late 2020, her father passed
away and she inherited a residential property located in Cheung Chau
("Cheung Chau Property"). The Cheung Chau Property was let out at a monthly rent of
HK$9,800 inclusive of rates and government rent throughout the year of assessment 2021/22.
On 30 May 2021, Betty pledged the Cheung Chau Property to a bank and obtained a
mortgage loan to acquire a car parking space in Shatin ("Shatin CPS") for investment
purpose. She could only let out the Shatin CPS from 1 February 2022 at a monthly rent of
HK$3,800.
In August 2021, Betty started her sole-proprietorship business, namely B's Art House, as a
painting instructor. She rented a studio in Kwun Tong from the landlord to conduct
the painting classes. Up to 31 March 2022, she made profits of HK$180,000 (after all
necessary tax adjustments) from the painting classes. Moreover, she found that there was
surplus space in her studio, thus she entered into a lease agreement to let part of the studio
to her friend at a monthly rent of HK$2,700 since 1 November 2021.
During the year of assessment 2021/22, Betty paid interest of HK$10,000 for the mortgage
loan (i.e. HK$1,000 × 10 months). She also paid government rent of HK$2,000 and HK$800
for the Cheung Chau Property and the Shatin CPS respectively (rates of both properties were
fully waived).

Required:

Compute the property tax and profits tax liabilities of Betty for the year of assessment 2021/22. Assume B's Art House is eligible and has elected for the two-tiered profits tax rates. Ignore provisional tax.
        '''
        
        print("开始测试 core 函数...")
        print(f"输入: {test_input}")
        print("-" * 50)
        
        try:
            result = await core(test_input)
            print("测试结果:")
            print(f"状态: {result['status']}")
            print(f"内容: {result['content']}")
        except Exception as e:
            print(f"测试出错: {e}")
        
        print("-" * 50)
        print("测试完成")
    
    # 运行测试
    asyncio.run(test_core())
