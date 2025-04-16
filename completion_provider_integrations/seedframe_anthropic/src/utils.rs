use regex::Regex;
use super::ContentBlock;

pub(crate) fn parse_content_blocks(input: &str) -> Vec<ContentBlock> {
    let re = Regex::new(r"(</?sf_(r_)?thinking>)|([^<]+)").unwrap();
    let mut result = Vec::new();
    let mut current_text = String::new();
    let mut current_block = None;
    let thinking_tag = ("<sf_thinking>", "</sf_thinking>");
    let redacted_thinking_tag = ("<sf_r_thinking>", "</sf_r_thinking>");
    let thinking_signature_tag = "/sf_sig>";

    for cap in re.captures_iter(input) {
        if let Some(tag) = cap.get(1) {
            let tag = tag.as_str();
            match &mut current_block {
                Some((is_redacted, content)) => {
                    let expected_end = if *is_redacted {
                        redacted_thinking_tag.1
                    } else {
                        thinking_tag.1
                    };

                    if tag == expected_end {
                        let content = std::mem::take(content);
                        result.push(if *is_redacted {
                            ContentBlock::RedactedThinking{ data: content }
                        } else {
                            let thinking = content.split(thinking_signature_tag).collect::<Vec<&str>>();
                            ContentBlock::Thinking{ thinking: thinking[0].to_string(), signature: thinking[1].to_string() }
                        });
                        current_block = None;
                    } else {
                        content.push_str(tag);
                    }
                }
                None => {
                    if tag == thinking_tag.0 {
                        if !current_text.is_empty() {
                            result.push(ContentBlock::Text{ text: std::mem::take(&mut current_text) });
                        }
                        current_block = Some((false, String::new()));
                    } else if tag == redacted_thinking_tag.0 {
                        if !current_text.is_empty() {
                            result.push(ContentBlock::Text{ text: std::mem::take(&mut current_text) });
                        }
                        current_block = Some((true, String::new()));
                    } else {
                        current_text.push_str(tag);
                    }
                }
            }
        } else if let Some(text) = cap.get(3) {
            let text = text.as_str();
            if let Some((_, content)) = &mut current_block {
                content.push_str(text);
            } else {
                current_text.push_str(text);
            }
        }
    }

    if let Some((is_redacted, content)) = current_block {
        let tag = if is_redacted {
            redacted_thinking_tag.0
        } else {
            thinking_tag.0
        };
        current_text.push_str(tag);
        current_text.push_str(&content);
    }

    if !current_text.is_empty() {
        result.push(ContentBlock::Text{ text: current_text });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_content_blocks() {
        let input = "hello world <sf_thinking> how are you</sf_sig>uuulala</sf_thinking> blabla<sf_r_thinking>ulalala</sf_r_thinking>";
        let result = parse_content_blocks(input);
        
        let expected = vec![
            ContentBlock::Text{ text: "hello world ".to_string() },
            ContentBlock::Thinking{ thinking: " how are you".to_string(), signature: "uuulala".to_string() },
            ContentBlock::Text{ text: " blabla".to_string() },
            ContentBlock::RedactedThinking{ data: "ulalala".to_string() },
        ];
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_no_tags() {
        let input = "just some regular text";
        let result = parse_content_blocks(input);
        
        let expected = vec![
            ContentBlock::Text{ text: "just some regular text".to_string() },
        ];
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiple_tags() {
        let input = "start<sf_thinking>think1</sf_sig>sig</sf_thinking>middle<sf_r_thinking>think2</sf_r_thinking>end";
        let result = parse_content_blocks(input);
        
        let expected = vec![
            ContentBlock::Text{ text: "start".to_string() },
            ContentBlock::Thinking{ thinking: "think1".to_string(), signature: "sig".to_string() },
            ContentBlock::Text{ text: "middle".to_string() },
            ContentBlock::RedactedThinking{ data: "think2".to_string() },
            ContentBlock::Text{ text: "end".to_string() },
        ];
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_unclosed_tag() {
        let input = "text<sf_thinking>unclosed";
        let result = parse_content_blocks(input);
        
        let expected = vec![
            ContentBlock::Text{ text: "text".to_string() },
            ContentBlock::Text{ text: "<sf_thinking>unclosed".to_string() },
        ];
        
        assert_eq!(result, expected);
    }
}
