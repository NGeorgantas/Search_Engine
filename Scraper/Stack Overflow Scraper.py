def individual_comment2(url):
    import requests
    from bs4 import BeautifulSoup
    stoplists=["Thanks for contributing an answer to Stack Overflow!","But avoid …","To learn more, see our tips on writing great answers."
              ,"Required, but never shown","To subscribe to this RSS feed, copy and paste this URL into your RSS reader."
              ,"By clicking “Post Your Answer”, you agree to our terms of service, privacy policy and cookie policy",
              "By clicking “Accept all cookies”, you agree Stack Exchange can store cookies on your device and disclose information in accordance with our Cookie Policy."
              ,"Your privacy"]
    question=[]
    response = requests.get(url)
    content = BeautifulSoup(response.text, 'lxml')
    for para in content.findAll('p'):
        comment=para.text.strip().replace('\n', ' ')
        if comment not in stoplists:
            #print(comment)
            question.append(para.text.strip().replace('\n', ' '))
    return " ".join(question[4:]).strip().replace('\n', ' ')


def web_srapper(pages,folder_path):
    from random import randint
    import time
    import json
    import requests
    from bs4 import BeautifulSoup

    # Base url
    start_url = 'https://stackoverflow.com/questions?tab=newest&page='

    # Loop over Stack Overflow questions' pages
    my_data={}
    my_questions=[]

    for page_num in range(1, pages):
        

        # get next page url
        url = start_url + str(page_num)

        # make HTTP GET request to the given url
        response = requests.get(url)

        # parse content
        content = BeautifulSoup(response.text, 'lxml')

        # extract question links
        links = content.findAll('a', {'class': 'question-hyperlink'})

        # extract question description
        description = content.findAll('div', {'class': 'excerpt'})

        print('\n\nURL:', url)


        #my_questions=[]
        # loop over Stack Overflow question list
        for index in range(0, len(description)):
            seconds=randint(1,4)
            time.sleep(seconds)
            comment=individual_comment2("https://stackoverflow.com"+links[index]['href'])
            question={'title':links[index].text,
                     'url':"https:/stackoverflow.com"+links[index]['href'],
                     'description':comment
                    }
            my_questions.append(question)
    my_data["All questions"]=my_questions
    f = open(folder_path+f"\sample_{pages}after_changes.json", "w")
    json.dump(my_data, f, indent=2)
    f.close()
    return my_data


path=r"C:\Users\Panos2\Desktop\Materiasl\ACG\Data Mining and Search Engines\Data Mining"            
x=web_srapper(100,path)

