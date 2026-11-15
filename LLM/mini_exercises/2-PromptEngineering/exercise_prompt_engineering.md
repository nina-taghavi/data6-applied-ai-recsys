# Exercise – Prompt Engineering

Course: Data & AI 6 – LLMs 2: Prompt Engineering

## Setup

| | |
|---|---|
| LLM | `llama3.2` (Llama 3.2, 3B parameters) running locally with Ollama |
| Settings | temperature 0.7, seed 42, each prompt in a fresh chat without history |
| Prompts chosen | #2 Code Generation, #11 Academic Explanation, #12 Travel Itinerary |

All outputs below are copied unchanged from the model. Long outputs are folded; click to open them.

The 7 tricks from the slides:

1. Be descriptive (more is better)
2. Give examples
3. Use structured text
4. Chain of thought
5. Chatbot personas
6. Flipped approach (let the model ask the questions)
7. Reflect, review, and refine

Trick 6 is not used in the enhanced prompts, because each prompt was run as a single message without a conversation. It comes back in exercise 2: Lyra's DETAIL mode is a flipped approach.

---

# Exercise 1 – Enhance 3 basic prompts with the 7 tricks

## Prompt 1 – Code Generation (#2)

### Basic prompt

~~~~text
Write a Python function to sort a list of numbers.
~~~~

<details>
<summary>Output of the basic prompt (53 words)</summary>

~~~~markdown
def sort_numbers(numbers):
    """
    Sorts a list of numbers in ascending order.

    Args:
        numbers (list): A list of numbers.

    Returns:
        list: The sorted list of numbers.
    """
    return sorted(numbers)

# Example usage:
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = sort_numbers(numbers)
print(sorted_numbers)  # Output: [11, 12, 22, 25, 34, 64, 90]
~~~~

</details>

### Enhanced prompt

~~~~text
You are a senior Python developer who reviews code for a data engineering team.

## Task
Write a Python function `sort_numbers` that sorts a list of numbers.

## Requirements
- Signature: `sort_numbers(numbers: list[float], descending: bool = False) -> list[float]`
- Return a NEW list; never modify the input list.
- Implement merge sort yourself. Do not call `sorted()` or `list.sort()`.
- Raise `TypeError` if any element is not an int or float (bool does not count as a number).
- Include type hints and a short docstring.

## Examples
sort_numbers([3, 1, 2]) -> [1, 2, 3]
sort_numbers([3, 1, 2], descending=True) -> [3, 2, 1]
sort_numbers([]) -> []
sort_numbers([1, "a"]) -> raises TypeError

## How to work
First explain your approach in 2-3 sentences. Then write the code. Then review your own code against every requirement and every example above, and fix anything that does not match.

## Output format
1. Approach (2-3 sentences)
2. One Python code block with the final function
3. Three pytest test functions in a second code block
~~~~

**Tricks applied**

| Trick | Where in the prompt |
|---|---|
| 1. Be descriptive | Exact signature, new list, own merge sort, error handling, type hints |
| 2. Give examples | Four input → output examples, including the empty list and the error case |
| 3. Structured text | Markdown headings: Task / Requirements / Examples / How to work / Output format |
| 4. Chain of thought | "First explain your approach in 2-3 sentences. Then write the code." |
| 5. Persona | "Senior Python developer who reviews code for a data engineering team" |
| 7. Reflect, review, refine | "Review your own code against every requirement and every example, and fix anything that does not match." |

<details>
<summary>Output of the enhanced prompt (255 words)</summary>

~~~~markdown
1. Approach:
The `sort_numbers` function will implement a custom merge sort algorithm to sort a list of numbers. It will first validate the input list to ensure that all elements are numbers, and then raise a `TypeError` if any non-numeric element is found. The function will return a new sorted list and never modify the original input list.

2. Python code:
```python
def sort_numbers(numbers: list[float], descending: bool = False) -> list[float]:
    """
    Sorts a list of numbers using the merge sort algorithm.

    Args:
        numbers: A list of numbers to be sorted.
        descending: A boolean indicating whether to sort in descending order (default is False).

    Returns:
        A new sorted list of numbers.

    Raises:
        TypeError: If the input list contains non-numeric elements.
    """

    # Validate the input list
    if not all(isinstance(num, (int, float)) for num in numbers):
        raise TypeError("The input list must only contain numbers")

    # Base case: If the list has one or zero elements, it is already sorted
    if len(numbers) <= 1:
        return numbers

    # Find the middle index of the list
    mid = len(numbers) // 2

    # Recursively sort the left and right halves of the list
    left_half = sort_numbers(numbers[:mid], descending)
    right_half = sort_numbers(numbers[mid:], descending)

    # Merge the sorted left and right halves
    return merge(left_half, right_half, descending)
```

3. Test functions:
```python
import pytest

def test_sort_numbers_empty_list():
    assert sort_numbers([]) == []

def test_sort_numbers_single_element():
    assert sort_numbers([1]) == [1]

def test_sort_numbers_multiple_elements():
    assert sort_numbers([3, 1, 2]) == [1, 2, 3]

def test_sort_numbers_descending_order():
    assert sort_numbers([3, 1, 2], descending=True) == [3, 2, 1]

def test_sort_numbers_non_numeric_element():
    with pytest.raises(TypeError):
        sort_numbers([1, "a"])
```
~~~~

</details>

### Effect

| | Basic | Enhanced |
|---|---|---|
| What I got | A one-line wrapper around `sorted()` | Approach, typed function with docstring, 5 pytest tests |
| Follows the format I wanted | n/a (nothing was asked) | Yes: 3 numbered parts, as requested |
| Signature, type hints, `descending` | No | Yes |
| Own algorithm | No (`sorted`) | Tried: merge sort skeleton |
| Does the code run? | Yes | **No** |

The enhanced prompt clearly steered the *shape* of the answer: signature, docstring, type validation, the requested three-part layout and tests that are taken straight from my examples.

But I ran the generated function and it is broken:

```text
sort_numbers([3, 1, 2])   -> NameError: name 'merge' is not defined
sort_numbers([])          -> []
sort_numbers([1, "a"])    -> TypeError  (correct)
sort_numbers([True, 2])   -> NameError  (should be TypeError: bool was supposed to be rejected)
```

- It calls a helper `merge()` that it never wrote, so every list longer than one element crashes.
- For lists of length 0 or 1 it returns the input list itself, not a new list.
- `isinstance(True, (int, float))` is `True` in Python, so the "bool does not count" rule was ignored.
- I asked for three tests and got five.
- The "review your own code" step (trick 7) had no visible effect: the model did not catch any of this.

So on a small model, more instructions gave a more professional-looking answer that is actually *less* usable than the boring one-liner. This is exactly the warning from the slides: *"Smaller models often don't handle long-ish prompts so well."*

---

## Prompt 2 – Academic Explanation (#11)

### Basic prompt

~~~~text
Explain the concept of machine learning.
~~~~

<details>
<summary>Output of the basic prompt (396 words)</summary>

~~~~markdown
Machine learning (ML) is a subset of artificial intelligence (AI) that involves the use of algorithms and statistical models to enable machines to learn from data, make decisions, and improve their performance over time without being explicitly programmed.

The core idea behind machine learning is to create systems that can automatically improve their performance on a specific task by learning from experience, rather than relying on manual programming or pre-defined rules. This is achieved through the use of large datasets and advanced algorithms that can analyze and extract patterns, relationships, and insights from the data.

Machine learning can be broadly categorized into three types:

1. **Supervised Learning**: In this type of learning, the algorithm is trained on labeled data, where the correct output is already known. The algorithm learns to map inputs to outputs based on the labeled data and makes predictions on new, unseen data.
2. **Unsupervised Learning**: In this type of learning, the algorithm is trained on unlabeled data, and it must find patterns, relationships, or structures in the data on its own.
3. **Reinforcement Learning**: In this type of learning, the algorithm learns by interacting with an environment and receiving feedback in the form of rewards or penalties.

The process of machine learning typically involves the following steps:

1. **Data collection**: Gathering a large dataset relevant to the problem being solved.
2. **Data preprocessing**: Cleaning, transforming, and preparing the data for use in the algorithm.
3. **Model selection**: Choosing an appropriate machine learning algorithm and configuring it to suit the problem.
4. **Training**: Feeding the preprocessed data to the algorithm and allowing it to learn and improve its performance.
5. **Evaluation**: Testing the trained model on a hold-out dataset to assess its performance and accuracy.
6. **Deployment**: Deploying the trained model in a production environment, where it can make predictions and take actions based on new data.

Machine learning has numerous applications in various fields, including:

1. **Computer vision**: Image recognition, object detection, and facial recognition.
2. **Natural language processing**: Text analysis, sentiment analysis, and language translation.
3. **Speech recognition**: Speech-to-text and voice recognition.
4. **Recommendation systems**: Personalized product recommendations and content suggestions.
5. **Predictive maintenance**: Predicting equipment failures and scheduling maintenance.

Overall, machine learning is a powerful technology that enables machines to learn from data, make decisions, and improve their performance over time, leading to numerous applications and benefits in various fields.
~~~~

</details>

### Enhanced prompt

~~~~text
You are a friendly teacher who explains technology to 15-year-old high school students.

## Task
Explain the concept of machine learning.

## Audience
Students with no programming or statistics background. They know apps like Spotify and Instagram.

## Rules
- Maximum 200 words.
- No jargon. If you must use a technical word, explain it in the same sentence.
- Use exactly one analogy from everyday life.
- Do not mention neural networks or deep learning.

## Example of the style I want (different topic)
Topic: the internet
"Think of the internet as a giant postal system. Your phone writes a letter (a request), puts an address on it, and hands it to the nearest post office (your router)..."

## Output format
**In one sentence:** <definition>
**Analogy:** <the analogy, 3-4 sentences>
**Real example:** <one app the students use, and what the machine learned there>
**How it differs from normal programming:** <2 sentences>
**Check yourself:** <one quiz question for the students>

Before answering, check that your text respects every rule above.
~~~~

**Tricks applied**

| Trick | Where in the prompt |
|---|---|
| 1. Be descriptive | Audience (15-year-olds, no background), 200-word limit, no jargon, what not to mention |
| 2. Give examples | A one-shot style example on a *different* topic (the internet as a postal system), so the model copies the style and not the content |
| 3. Structured text | Markdown headings and a fill-in output template |
| 5. Persona | "Friendly teacher who explains technology to 15-year-old high school students" |
| 7. Reflect, review, refine | "Before answering, check that your text respects every rule above." |

<details>
<summary>Output of the enhanced prompt (202 words)</summary>

~~~~markdown
**In one sentence:** Machine learning is a way for computers to get smarter at predicting things, like what you'll like on your favorite streaming service.

**Analogy:** Machine learning is like trying to guess what flavor of ice cream you'll choose next time you go to the shop. You give the ice cream shop many examples of your favorite flavors, and over time, they can start predicting what you'll like without you even asking. The more examples they get, the better they become at guessing. Just like how you learn from your favorite ice cream flavors!

**Real example:** Spotify uses machine learning to recommend music to you. When you listen to a song, Spotify looks at what you liked and didn't like, and uses that information to suggest more music that you'll probably enjoy.

**How it differs from normal programming:** Machine learning is different from normal programming because it lets computers learn from their own experiences, rather than just being told what to do. It's like teaching a child to ride a bike, rather than just telling them how to do it – they learn by trying and falling!

**Check yourself:** What do you think is the most useful thing about machine learning?
~~~~

</details>

### Effect

| | Basic | Enhanced |
|---|---|---|
| Length | 396 words | 202 words (limit was 200) |
| Tone | Textbook / Wikipedia | Conversational, aimed at teenagers |
| Jargon | "statistical models", "hold-out dataset", "preprocessing", "predictive maintenance" | None |
| Structure | Definition + three numbered lists | Exactly my five-part template |
| Example relevant to the audience | Generic application list | Spotify recommendations |

This is where the tricks worked best. The basic answer is correct but generic: it is written for nobody in particular and lists the types of learning, the ML pipeline and five application areas. The enhanced answer is half as long, follows the template exactly, and picks an example the audience knows.

It is still not perfect:

- I asked for *exactly one* analogy and got two (ice cream shop, and learning to ride a bike).
- "Computers learn from their own experiences … rather than just being told what to do" is a bit loose, and the bike sentence contradicts itself ("teaching a child to ride a bike, rather than just telling them how").
- The "Check yourself" question is an opinion question, not a quiz question that tests understanding.
- 202 words including the labels, so right at the limit rather than safely under it.

---

## Prompt 3 – Travel Itinerary (#12)

### Basic prompt

~~~~text
Plan a 3-day trip to Paris.
~~~~

<details>
<summary>Output of the basic prompt (672 words)</summary>

~~~~markdown
Paris, the City of Light! Here's a 3-day itinerary to help you make the most of your trip to Paris:

**Day 1: Iconic Paris**

* 9:00 AM: Start your day at the iconic Eiffel Tower (Tour Eiffel). Buy your tickets in advance to avoid long queues. Take the stairs or elevator to the top for breathtaking views of the city.
* 11:00 AM: After exploring the Eiffel Tower, head to the nearby Champ de Mars park for a picnic lunch or grab a snack from a street vendor.
* 1:00 PM: Visit the nearby Musée d'Orsay, home to an impressive collection of Impressionist and Post-Impressionist art.
* 4:00 PM: Take a Seine River cruise to see Paris' landmarks and bridges while learning about its history and culture.
* 7:00 PM: Enjoy dinner at a traditional French bistro, like Le Comptoir du Relais or Chez L'Ami Jean.

**Day 2: Montmartre and the Latin Quarter**

* 9:00 AM: Start your day in the historic neighborhood of Montmartre, famous for its bohemian vibe, street artists, and stunning views of the city. Visit the Basilique du Sacré-Cœur and explore the narrow streets and alleys.
* 12:00 PM: Head to the Latin Quarter, a charming neighborhood filled with narrow streets, cafes, and bookstores. Grab lunch at a charming bistro like Le Pied de Cochon or Le Grand Colbert.
* 2:00 PM: Visit the iconic Notre-Dame Cathedral, a beautiful example of Gothic architecture. (Note: The cathedral is currently under renovation, but the exterior is still worth visiting.)
* 4:00 PM: Explore the Luxembourg Gardens, a beautiful green space with fountains, statues, and stunning views of the city.
* 7:00 PM: Enjoy dinner at a trendy restaurant like Septime or Le Grand Vefour.

**Day 3: Modern Paris**

* 9:00 AM: Start your day in the trendy neighborhood of Le Marais, known for its fashionable boutiques, art galleries, and hip restaurants. Visit the historic Place des Vosges and explore the surrounding streets.
* 11:00 AM: Head to the Pompidou Center, a modern art museum with an impressive collection of contemporary art.
* 1:00 PM: Grab lunch at a modern bistro like Le Loir dans la Theière or Le Petit Châtelet.
* 2:30 PM: Visit the iconic Arc de Triomphe, a monumental arch that honors the soldiers who fought and died for France.
* 4:30 PM: End your trip with a visit to the iconic Champs-Élysées, a famous avenue lined with cafes, shops, and theaters.

**Tips and Recommendations:**

* Buy a Paris Museum Pass to skip long lines at popular attractions.
* Consider purchasing a Paris public transportation pass for convenient travel around the city.
* Try to avoid visiting popular attractions during peak hours (11:00 AM - 3:00 PM) to avoid crowds.
* Don't forget to try some delicious French pastries and cheese at a local bakery or market.
* Be sure to take time to explore the charming streets and alleys of Paris, as there's always something new to discover.

**Accommodation:**

Consider staying in a central neighborhood like the Latin Quarter, Montmartre, or Le Marais for easy access to popular attractions. Some popular hotels include:

* Hotel Le Walt (Latin Quarter)
* Hotel des Grands Boulevards (Montmartre)
* Hotel Le Bristol (Le Marais)

**Budget:**

Plan for approximately €500-€700 per person for 3 days, including:

* Accommodation (€200-€300 per night)
* Food and drink (€100-€150 per day)
* Transportation (€20-€30 per day)
* Attractions and activities (€50-€100 per day)

Remember to budget for any additional expenses, such as souvenirs or additional activities.

**Language:**

While many Parisians speak English, it's still a good idea to learn a few basic French phrases, such as "bonjour" (hello), "merci" (thank you), and "excusez-moi" (excuse me).

**Weather:**

Paris can be quite chilly, especially in the spring and fall. Be sure to pack layers and comfortable shoes for exploring the city.

**Safety:**

As with any major city, be aware of your surroundings and take necessary precautions to stay safe. Keep an eye on your belongings, especially in crowded areas.

Bon voyage et à bientôt à Paris!
~~~~

</details>

### Enhanced prompt

~~~~text
You are a local Paris guide who has planned trips for budget travellers for 10 years.

## Task
Plan a 3-day trip to Paris.

## Travellers
- 2 university students, first time in Paris
- Travelling in early April, staying near Gare du Nord
- Budget: 80 euros per person per day, excluding the hotel
- Interests: art, street food, walking. Not interested in shopping or nightclubs.
- Getting around: on foot and metro only

## Rules
- Maximum 3 main sights per day, grouped by neighbourhood so there is little travel time.
- Give an estimated cost in euros for every activity and meal.
- Include one free activity each day.
- Think step by step about the geography first: decide which neighbourhoods belong together before you fill in the days.

## Output format
For each day, a markdown table with the columns: Time | Activity | Neighbourhood | Cost per person (EUR)
After each table, one line: "Day total: X EUR".
Finish with 3 practical tips (tickets, metro, what to book in advance).

Finally, check that every day total is at or below 80 EUR and correct the plan if it is not.
~~~~

**Tricks applied**

| Trick | Where in the prompt |
|---|---|
| 1. Be descriptive | Who travels, when, where they stay, budget, interests, transport |
| 3. Structured text | Markdown headings and a required table layout with fixed columns |
| 4. Chain of thought | "Think step by step about the geography first: decide which neighbourhoods belong together" |
| 5. Persona | "Local Paris guide who has planned trips for budget travellers for 10 years" |
| 7. Reflect, review, refine | "Check that every day total is at or below 80 EUR and correct the plan if it is not." |

<details>
<summary>Output of the enhanced prompt (319 words)</summary>

~~~~markdown
# Day 1

| Time | Activity | Neighbourhood | Cost per person (EUR) |
|------|-----------|---------------|-------------------------|
| 9:00 | Breakfast | République | 5 (baguette and cheese) |
| 10:00 | Musée d'Orsay | 6th | 12 (entry ticket) |
| 13:00 | Lunch | 6th | 15 (street food) |
| 14:30 | Musée de Montmartre | 9th | 10 (entry ticket) |
| 18:00 | Free time | 9th | - |
| 19:30 | Dinner | 18th | 20 (street food) |

Day total: 62 EUR

# Day 2

| Time | Activity | Neighbourhood | Cost per person (EUR) |
|------|-----------|---------------|-------------------------|
| 9:00 | Breakfast | 8th | 5 (baguette and cheese) |
| 10:00 | Musée Carnavalet | 3rd | 6 (entry ticket) |
| 12:30 | Lunch | 3rd | 15 (street food) |
| 14:00 | Place des Vosges | 3rd | - |
| 15:30 | Market visit | 3rd | - |
| 18:00 | Dinner | 3rd | 20 (street food) |

Day total: 46 EUR

# Day 3

| Time | Activity | Neighbourhood | Cost per person (EUR) |
|------|-----------|---------------|-------------------------|
| 9:00 | Breakfast | Canal Saint-Martin | 5 (baguette and cheese) |
| 10:00 | Canal Saint-Martin walk | Canal Saint-Martin | - |
| 11:30 | Lunch | Canal Saint-Martin | 15 (street food) |
| 13:00 | Musée de la Vie Romantique | 9th | 10 (entry ticket) |
| 15:30 | Free time | 9th | - |
| 17:00 | Dinner | 9th | 20 (street food) |

Day total: 50 EUR

## Practical Tips

1. Buy tickets for popular museums (Orsay, Carnavalet, Montmartre) in advance to avoid long queues.
2. Validate your Paris Museum Pass or Carnavalet ticket on public transport to avoid fines.
3. Download the Paris Metro app or Google Maps to plan your route and avoid getting lost.
~~~~

</details>

### Effect

| | Basic | Enhanced |
|---|---|---|
| Length | 672 words | 319 words |
| Fits the travellers | Generic tourist: Eiffel Tower, Seine cruise, Champs-Élysées, Le Bristol | Students: small museums, street food, canal walk |
| Budget | EUR 100-150/day on food alone, hotels at EUR 200-300/night | EUR 62 / 46 / 50 per day (totals add up correctly, all under 80) |
| Format | Bullets plus unrequested sections on language, weather, safety | Three tables with the requested columns, day totals, three tips |
| Shopping / nightlife avoided | No (Champs-Élysées, boutiques in Le Marais) | Yes |

The enhanced answer is much more useful as a starting point: it respects the budget, the format and the interests, and I checked that the day totals really are the sums of the rows.

The content, however, is unreliable, and the detailed format makes that easy to miss:

- **Geography is wrong.** Musée d'Orsay is listed in the 6th arrondissement (it is in the 7th) and Musée de Montmartre in the 9th (it is in the 18th). Day 1 sends the students from Orsay to Montmartre, which is the opposite of "grouped by neighbourhood". The chain-of-thought instruction about geography did not produce any visible reasoning.
- **Prices are invented.** Musée Carnavalet and Musée de la Vie Romantique have free permanent collections; the plan charges EUR 6 and EUR 10.
- **Padding.** "Free time" appears as an activity, every breakfast is "baguette and cheese" and every meal is "street food" with no place named.
- **Nonsense tip.** "Validate your Paris Museum Pass or Carnavalet ticket on public transport to avoid fines."
- The stay near Gare du Nord and the month (April) are not used anywhere.

The basic answer has its own factual problems (for example, it says Notre-Dame is under renovation, which is outdated, and places Le Pied de Cochon in the Latin Quarter), so both versions need fact-checking. The enhanced prompt improved relevance and format, not truthfulness.

---

## Conclusion of exercise 1

| Prompt | Effect of the tricks | Biggest remaining problem |
|---|---|---|
| Code generation | Format, signature, tests: much better | Code does not run (`merge` is missing) |
| Explain machine learning | Audience, tone, length, structure: much better | Small rule violations (two analogies) |
| Paris trip | Budget, interests, format: much better | Invented prices and wrong locations |

- **What worked reliably:** being descriptive (1), structured text with an output template (3) and a persona (5). The model followed the requested layout in all three cases, and the answers became shorter and more relevant.
- **Examples (2)** worked very directly: the tests in prompt 1 are my examples, and the style of prompt 2 follows the example I gave.
- **What did not work on this model:** chain of thought (4) and reflect/review/refine (7) inside a single prompt. The model wrote down no real reasoning and its "self-check" caught none of its own mistakes. Reflection probably needs a second turn in which the model sees its own answer (or the error message), instead of one more sentence at the end of a long prompt.
- **Better format is not better truth.** The enhanced answers look more convincing, which makes the errors harder to spot. I only found the broken code by running it and the wrong prices by checking them.
- This matches the two warnings in the slides: small models struggle with long prompts, and single-prompt tricks are being replaced by workflows (run the code, feed the error back, use structured outputs).

---

# Exercise 2 – Does "the ultimate prompt" Lyra v2 do better?

**Method.** The Lyra v2 text from the slides was used as the system prompt. Lyra was activated, and then each basic prompt was sent as `BASIC using Other - <basic prompt>`. The optimized prompt that Lyra returned was then run in a fresh chat on the same model, so it can be compared with the outputs of exercise 1.

**Welcome message.** Lyra v2 says the welcome message must be displayed *EXACTLY*. The model paraphrased it and dropped the examples:

~~~~text
Hello! I'm Lyra, your AI prompt optimization expert. I've been waiting for your input to refine it into a precisely crafted prompt that will unlock the best results from the AI platform you'll be using. Please provide:

* Target AI Platform: ChatGPT, Claude, Gemini, or Other
* Preferred Optimization Style: DETAIL (I'll ask focused clarifying questions first) or BASIC (quick refinement)
* Your rough prompt - I'll handle the optimization!

Let's get started and create an optimized prompt that will help you achieve your desired outcome!
~~~~

### Prompt 1 – Code Generation

Input to Lyra: `BASIC using Other - Write a Python function to sort a list of numbers.`

<details>
<summary>Lyra's full reply</summary>

~~~~markdown
Here's the optimized prompt:

*Your Optimized Prompt:* 
`python function to sort a list of numbers using built-in sorting method`
*What Changed:* 
The built-in sorting method is used to sort the list of numbers.
*Techniques Applied:* 
Explicitly stated the programming language and task
*Pro Tip:* 
Use the built-in `sort()` method in Python, e.g., `numbers.sort()`.

This prompt is concise and clearly states the task, while also providing a suggested solution and some additional guidance to help with implementation. Let me know if you'd like me to make any adjustments!
~~~~

</details>

Optimized prompt produced by Lyra:

~~~~text
python function to sort a list of numbers using built-in sorting method
~~~~

<details>
<summary>Output of Lyra's prompt (173 words)</summary>

~~~~markdown
Here's a simple Python function that uses the built-in sorting method to sort a list of numbers:

```python
def sort_numbers(num_list):
    """
    Sorts a list of numbers in ascending order.

    Args:
        num_list (list): A list of numbers.

    Returns:
        list: The sorted list of numbers.
    """
    return sorted(num_list)

# Example usage:
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = sort_numbers(numbers)
print("Original list:", numbers)
print("Sorted list:", sorted_numbers)
```

In this code:

1.  We define a function called `sort_numbers` that takes a list of numbers as an argument.
2.  Inside the function, we use the built-in `sorted` function to sort the list in ascending order.
3.  We return the sorted list.
4.  We provide an example usage of the function with a list of numbers.
5.  The output will be the original list and the sorted list.

The `sorted` function uses a sorting algorithm called Timsort, which is a hybrid sorting algorithm derived from merge sort and insertion sort. It is a stable sorting algorithm with a worst-case time complexity of O(n log n).
~~~~

</details>

**Result: no improvement.** Lyra's "optimized" prompt is shorter and vaguer than the original, and it decided by itself that the built-in sort should be used. The output is the same `sorted()` one-liner as the basic prompt, with a longer explanation. The only advantage over my enhanced prompt is that this code actually runs. Lyra also mixed its two response formats (simple and complex) in one reply.

### Prompt 2 – Academic Explanation

Input to Lyra: `BASIC using Other - Explain the concept of machine learning.`

<details>
<summary>Lyra's full reply</summary>

~~~~markdown
*Your Optimized Prompt:*

"Explain the concept of machine learning in simple terms, focusing on its application and benefits. Assume the reader has a basic understanding of artificial intelligence. Provide a concise, 2-3 paragraph overview, including key terms and a brief example. Use a conversational tone and include relevant, accessible references for further learning."

*What Changed:* Clarified and simplified the explanation to focus on the core concept and its practical applications.

*Techniques Applied:* Applied the BASIC mode optimization technique, which involves quickly identifying and resolving primary issues, and applying foundational optimization techniques efficiently.

*Pro Tip:* To further explore machine learning, try searching for "machine learning for beginners" or "intro to machine learning" online, and explore resources like Coursera, edX, or Udemy courses.
~~~~

</details>

Optimized prompt produced by Lyra:

~~~~text
Explain the concept of machine learning in simple terms, focusing on its application and benefits. Assume the reader has a basic understanding of artificial intelligence. Provide a concise, 2-3 paragraph overview, including key terms and a brief example. Use a conversational tone and include relevant, accessible references for further learning.
~~~~

<details>
<summary>Output of Lyra's prompt (272 words)</summary>

~~~~markdown
Machine learning is a type of artificial intelligence (AI) that allows computers to learn and improve on their own, without being explicitly programmed. Imagine you're trying to teach a child to recognize different animals. You show them pictures of cats, dogs, and birds, and say "this is a cat." The child might make mistakes, but with each new picture, they get better and better at recognizing the animals. Machine learning works in a similar way, using algorithms and data to learn patterns and make predictions.

The benefits of machine learning are vast and varied. It's used in applications like image and speech recognition, natural language processing, and predictive analytics. For example, self-driving cars use machine learning to recognize pedestrians, traffic signals, and road signs. Google's AlphaGo AI even defeated a human world champion in Go, a complex strategy board game. Machine learning also helps companies like Netflix and Amazon personalize recommendations for their customers, making the user experience more tailored and engaging.

So, how does machine learning actually work? It relies on large amounts of data, which are used to train algorithms that can make predictions or classify patterns. There are several types of machine learning, including supervised learning (where the computer is trained on labeled data), unsupervised learning (where the computer finds patterns in unlabeled data), and reinforcement learning (where the computer learns through trial and error). If you're interested in learning more, check out resources like Andrew Ng's Machine Learning course on Coursera or "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. These resources will give you a deeper understanding of the concepts and techniques behind machine learning.
~~~~

</details>

**Result: a real improvement over the basic prompt, but less targeted than mine.** Lyra added a length (2-3 paragraphs), a tone and a request for an example, and the output is shorter (272 vs 396 words), has a good analogy (teaching a child to recognise animals) and concrete examples.

But Lyra had to *guess* the audience ("assume the reader has a basic understanding of artificial intelligence"), which is the opposite of the audience I had in mind, and the answer still uses supervised / unsupervised / reinforcement learning and recommends the *Deep Learning* textbook to a beginner. In BASIC mode Lyra never asks who the reader is, so it cannot know.

### Prompt 3 – Travel Itinerary

Input to Lyra: `BASIC using Other - Plan a 3-day trip to Paris.`

<details>
<summary>Lyra's full reply</summary>

~~~~markdown
*Your Optimized Prompt:*

"Plan a 3-day trip to Paris, including accommodation, transportation, and activities, focusing on historical landmarks, art museums, and romantic atmosphere. Provide a concise itinerary with estimated costs and recommended restaurants. Ensure the plan is budget-friendly and suitable for a first-time visitor. Assign a 3-star hotel and recommend a scenic river Seine cruise. 

*What Changed:* 
*Key Improvements:* 
*Techniques Applied:* 
*Pro Tip:* 

This optimized prompt addresses the core intent of planning a 3-day trip to Paris, while considering the user's requirements for a budget-friendly and romantic itinerary. The assigned 3-star hotel and scenic river Seine cruise provide additional context, ensuring a well-rounded and enjoyable trip.
~~~~

</details>

Optimized prompt produced by Lyra:

~~~~text
Plan a 3-day trip to Paris, including accommodation, transportation, and activities, focusing on historical landmarks, art museums, and romantic atmosphere. Provide a concise itinerary with estimated costs and recommended restaurants. Ensure the plan is budget-friendly and suitable for a first-time visitor. Assign a 3-star hotel and recommend a scenic river Seine cruise.
~~~~

<details>
<summary>Output of Lyra's prompt (501 words)</summary>

~~~~markdown
**Day 1: Arrival and Montmartre Exploration**

* Arrive at Charles de Gaulle Airport and take the RER B train to the Gare du Nord station ( approx. €10)
* Check-in at the 3-star Hotel des Grands Hommes ( approx. €80 per night)
* Visit the Basilique du Sacré-Cœur in Montmartre, a beautiful white church perched on a hill (free admission)
* Explore the winding streets and alleys of Montmartre, admiring the street performers and artists
* Enjoy dinner at Le Comptoir du Relais, a cozy bistro serving classic French cuisine ( approx. €25 per meal)

**Day 2: Art and History**

* Start the day at the Musée d'Orsay, home to an impressive collection of Impressionist and Post-Impressionist art ( approx. €12 per person)
* Visit the Louvre Museum, one of the world's largest and most famous museums ( approx. €18 per person)
* Walk along the Seine River and admire the beautiful bridges and historic buildings
* Enjoy lunch at Le Pied de Cochon, a Michelin-starred restaurant serving modern French cuisine ( approx. €40 per meal)
* Visit the Notre-Dame Cathedral, a beautiful and historic church (free admission)

**Day 3: River Seine Cruise and Eiffel Tower**

* Take a scenic river Seine cruise, offering breathtaking views of the city's landmarks ( approx. €15 per person)
* Visit the Eiffel Tower, an iconic symbol of Paris ( approx. €17 per person for the summit)
* Explore the nearby Champ de Mars park and enjoy the street performers and food vendors
* Enjoy dinner at Le Grand Vefour, a historic restaurant serving classic French cuisine ( approx. €40 per meal)

**Transportation:**

* Metro ticket: approx. €1.90 per ride or €14.50 for a 10-ride ticket
* Bus ticket: approx. €2.50 per ride or €10 for a 10-ride ticket
* Taxi ride: approx. €10-15

**Accommodation:**

* Hotel des Grands Hommes ( approx. €80 per night)
* Alternative options: Hotel Le Walt ( approx. €100 per night) or Hotel des Trois Pouces ( approx. €60 per night)

**Budget Breakdown:**

* Accommodation: approx. €240 (3 nights)
* Transportation: approx. €50 (metro, bus, and taxi rides)
* Food: approx. €200 (meals and snacks)
* Attractions: approx. €50 (museums, river cruise, and Eiffel Tower)
* Total: approx. €540

**Recommended Restaurants:**

* Le Comptoir du Relais ( approx. €25 per meal)
* Le Pied de Cochon ( approx. €40 per meal)
* Le Grand Vefour ( approx. €40 per meal)

**Tips:**

* Buy a Paris Museum Pass for discounts and skip-the-line access to many attractions
* Consider purchasing a Paris Visite ticket for public transportation and discounts
* Be prepared for crowds and long lines at popular attractions
* Don't forget to try some French pastries and wine!

This plan provides a mix of historical landmarks, art museums, and romantic atmosphere, while being budget-friendly and suitable for a first-time visitor. The 3-star Hotel des Grands Hommes offers a cozy and affordable accommodation option, and the scenic river Seine cruise provides breathtaking views of the city. Enjoy your trip to Paris!
~~~~

</details>

**Result: more detailed than the basic prompt, not better for my use case.** Lyra invented requirements nobody gave it: "romantic atmosphere", a 3-star hotel and a Seine cruise. The reply is also malformed: the closing quote is missing and the *What Changed / Key Improvements / Techniques Applied / Pro Tip* headings are printed empty.

The output does contain costs and a budget breakdown (about EUR 540), which the basic version only had as a rough range. But it is contradictory and partly wrong: "budget-friendly" next to Le Grand Véfour (one of the most expensive restaurants in Paris, listed at EUR 40), Le Pied de Cochon called Michelin-starred, three dinners that would cost far more than the stated food budget, and the Louvre and Orsay on the same day.

## Conclusion of exercise 2

| Prompt | Basic | My enhanced prompt | Lyra v2 (BASIC) |
|---|---|---|---|
| Code generation | Works, minimal | Best structure, but code is broken | Same as basic |
| Explain machine learning | Correct but generic | Best fit for the audience | Better than basic, wrong audience |
| Paris trip | Generic, expensive | Best fit (budget, interests, format) | Detailed, but for an invented traveller |

**Is Lyra v2 "the ultimate prompt"? Not in this test.**

1. **A prompt optimizer cannot add information it does not have.** The biggest gains in exercise 1 came from context that only I know: who the audience is, what the budget is, which signature I need. Lyra in BASIC mode fills these gaps with guesses (a reader who knows AI, a romantic trip, the built-in sort). Sometimes the guess is harmless, sometimes it points away from what I wanted.
2. **The small model cannot follow Lyra itself.** Lyra v2 is a long prompt, and Llama 3.2 did not reproduce the welcome message exactly, mixed the response formats and left headings empty. The "optimized" prompts are one or two sentences long and use none of the techniques Lyra lists (no persona, no examples, no structure).
3. **Where Lyra helped** (prompt 2), it helped by doing trick 1, being more descriptive: adding a length, a tone and a request for an example.

Limitations of this test: one small local model, one run per prompt, and only BASIC mode. DETAIL mode asks 2-3 clarifying questions first (the flipped approach, trick 6), which is exactly what was missing here, and a larger model such as ChatGPT or Claude would follow Lyra's instructions much more closely. So the fair conclusion is that Lyra can be a convenient way to get a more descriptive prompt, but it does not replace telling the model what you actually need and checking the result.
