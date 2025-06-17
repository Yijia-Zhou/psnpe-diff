package com.qualcomm.qti.psnpedemo.post;

import com.google.common.base.Ascii;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Ints;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.google.common.base.Verify.verify;

public class MobileBertUtil {
    private final int MAX_ANS_LEN = 32;
    private final int MAX_QUERY_LEN = 64;
    public static final int MAX_SEQ_LEN = 384;
    private final boolean DO_LOWER_CASE = true;
    private final int PREDICT_ANS_NUM = 5;
    private static final int OUTPUT_OFFSET = 1;
    private static final Joiner SPACE_JOINER = Joiner.on(" ");
    private final int NUM_LITE_THREADS = 4;
    private final String IDS_TENSOR_NAME = "ids";
    private final String MASK_TENSOR_NAME = "mask";
    private final String SEGMENT_IDS_TENSOR_NAME = "segment_ids";
    private final String END_LOGITS_TENSOR_NAME = "end_logits";
    private final String START_LOGITS_TENSOR_NAME = "start_logits";
    private Map<String, Integer> dic = new HashMap<>();
    private FeatureConverter featureConverter;
    public MobileBertUtil(String vocabPath) throws Exception{
        Map<String, Integer> loadedDic;
        FileInputStream vocabFileStream = new FileInputStream(
                new File(vocabPath));
        loadedDic = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(vocabFileStream))) {
            int index = 0;
            while (reader.ready()) {
                String key = reader.readLine();
                loadedDic.put(key, index++);
            }
        }
        verify(loadedDic != null, "dic can't be null.");
        dic.putAll(loadedDic);
        featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN);
    }

    public Feature getFeature(String query, String content){
        return featureConverter.convert(query, content);
    }

    public synchronized List<QaAnswer> getBestAnswers(
            float[] startLogits, float[] endLogits, Feature feature) {
        // Model uses the closed interval [start, end] for indices.
        int[] startIndexes = getBestIndex(startLogits);
        int[] endIndexes = getBestIndex(endLogits);

        List<QaAnswer.Pos> origResults = new ArrayList<>();
        for (int start : startIndexes) {
            for (int end : endIndexes) {
                if (!feature.tokenToOrigMap.containsKey(start + OUTPUT_OFFSET)) {
                    continue;
                }
                if (!feature.tokenToOrigMap.containsKey(end + OUTPUT_OFFSET)) {
                    continue;
                }
                if (end < start) {
                    continue;
                }
                int length = end - start + 1;
                if (length > MAX_ANS_LEN) {
                    continue;
                }
                origResults.add(new QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]));
            }
        }

        Collections.sort(origResults);

        List<QaAnswer> answers = new ArrayList<>();
        for (int i = 0; i < origResults.size(); i++) {
            if (i >= PREDICT_ANS_NUM) {
                break;
            }

            String convertedText;
            if (origResults.get(i).start > 0) {
                convertedText = convertBack(feature, origResults.get(i).start, origResults.get(i).end);
            } else {
                convertedText = "";
            }
            QaAnswer ans = new QaAnswer(convertedText, origResults.get(i));
            answers.add(ans);
        }
        return answers;
    }

    /** Get the n-best logits from a list of all the logits. */
    private synchronized int[] getBestIndex(float[] logits) {
        List<QaAnswer.Pos> tmpList = new ArrayList<>();
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            tmpList.add(new QaAnswer.Pos(i, i, logits[i]));
        }
        Collections.sort(tmpList);

        int[] indexes = new int[PREDICT_ANS_NUM];
        for (int i = 0; i < PREDICT_ANS_NUM; i++) {
            indexes[i] = tmpList.get(i).start;
        }

        return indexes;
    }

    /** Convert the answer back to original text form. */
    private static String convertBack(Feature feature, int start, int end) {
        // Shifted index is: index of logits + offset.
        int shiftedStart = start + OUTPUT_OFFSET;
        int shiftedEnd = end + OUTPUT_OFFSET;
        int startIndex = feature.tokenToOrigMap.get(shiftedStart);
        int endIndex = feature.tokenToOrigMap.get(shiftedEnd);
        // end + 1 for the closed interval.
        String ans = SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1));
        return ans;
    }

   static public class QaAnswer {
        public Pos pos;
        public String text;

        public QaAnswer(String text, Pos pos) {
            this.text = text;
            this.pos = pos;
        }

        public QaAnswer(String text, int start, int end, float logit) {
            this(text, new Pos(start, end, logit));
        }

        /** Position and related information from the model. */
        public static class Pos implements Comparable<Pos> {
            public int start;
            public int end;
            public float logit;

            public Pos(int start, int end, float logit) {
                this.start = start;
                this.end = end;
                this.logit = logit;
            }

            @Override
            public int compareTo(Pos other) {
                return Float.compare(other.logit, this.logit);
            }
        }
    }

    public final class Feature {
        public final int[] inputIds;
        public final int[] inputMask;
        public final int[] segmentIds;
        public final List<String> origTokens;
        public final Map<Integer, Integer> tokenToOrigMap;

        public Feature(
                List<Integer> inputIds,
                List<Integer> inputMask,
                List<Integer> segmentIds,
                List<String> origTokens,
                Map<Integer, Integer> tokenToOrigMap) {
            this.inputIds = Ints.toArray(inputIds);
            this.inputMask = Ints.toArray(inputMask);
            this.segmentIds = Ints.toArray(segmentIds);
            this.origTokens = origTokens;
            this.tokenToOrigMap = tokenToOrigMap;
        }
    }


    public final class FeatureConverter {
        private final FullTokenizer tokenizer;
        private final int maxQueryLen;
        private final int maxSeqLen;

        public FeatureConverter(
                Map<String, Integer> inputDic, boolean doLowerCase, int maxQueryLen, int maxSeqLen) {
            this.tokenizer = new FullTokenizer(inputDic, doLowerCase);
            this.maxQueryLen = maxQueryLen;
            this.maxSeqLen = maxSeqLen;
        }

        public Feature convert(String query, String context) {
            List<String> queryTokens = tokenizer.tokenize(query);
            if (queryTokens.size() > maxQueryLen) {
                queryTokens = queryTokens.subList(0, maxQueryLen);
            }

            List<String> origTokens = Arrays.asList(context.trim().split("\\s+"));
            List<Integer> tokenToOrigIndex = new ArrayList<>();
            List<String> allDocTokens = new ArrayList<>();
            for (int i = 0; i < origTokens.size(); i++) {
                String token = origTokens.get(i);
                List<String> subTokens = tokenizer.tokenize(token);
                for (String subToken : subTokens) {
                    tokenToOrigIndex.add(i);
                    allDocTokens.add(subToken);
                }
            }

            // -3 accounts for [CLS], [SEP] and [SEP].
            int maxContextLen = maxSeqLen - queryTokens.size() - 3;
            if (allDocTokens.size() > maxContextLen) {
                allDocTokens = allDocTokens.subList(0, maxContextLen);
            }

            List<String> tokens = new ArrayList<>();
            List<Integer> segmentIds = new ArrayList<>();

            // Map token index to original index (in feature.origTokens).
            Map<Integer, Integer> tokenToOrigMap = new HashMap<>();

            // Start of generating the features.
            tokens.add("[CLS]");
            segmentIds.add(0);

            // For query input.
            for (String queryToken : queryTokens) {
                tokens.add(queryToken);
                segmentIds.add(0);
            }

            // For Separation.
            tokens.add("[SEP]");
            segmentIds.add(0);

            // For Text Input.
            for (int i = 0; i < allDocTokens.size(); i++) {
                String docToken = allDocTokens.get(i);
                tokens.add(docToken);
                segmentIds.add(1);
                tokenToOrigMap.put(tokens.size(), tokenToOrigIndex.get(i));
            }

            // For ending mark.
            tokens.add("[SEP]");
            segmentIds.add(1);

            List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);
            List<Integer> inputMask = new ArrayList<>(Collections.nCopies(inputIds.size(), 1));

            while (inputIds.size() < maxSeqLen) {
                inputIds.add(0);
                inputMask.add(0);
                segmentIds.add(0);
            }

            return new Feature(inputIds, inputMask, segmentIds, origTokens, tokenToOrigMap);
        }
    }
    static public final class BasicTokenizer {
        private final boolean doLowerCase;

        public BasicTokenizer(boolean doLowerCase) {
            this.doLowerCase = doLowerCase;
        }

        public List<String> tokenize(String text) {
            String cleanedText = cleanText(text);

            List<String> origTokens = whitespaceTokenize(cleanedText);

            StringBuilder stringBuilder = new StringBuilder();
            for (String token : origTokens) {
                if (doLowerCase) {
                    token = Ascii.toLowerCase(token);
                }
                List<String> list = runSplitOnPunc(token);
                for (String subToken : list) {
                    stringBuilder.append(subToken).append(" ");
                }
            }
            return whitespaceTokenize(stringBuilder.toString());
        }

        /* Performs invalid character removal and whitespace cleanup on text. */
        static String cleanText(String text) {
            if (text == null) {
                throw new NullPointerException("The input String is null.");
            }

            StringBuilder stringBuilder = new StringBuilder("");
            for (int index = 0; index < text.length(); index++) {
                char ch = text.charAt(index);

                // Skip the characters that cannot be used.
                if (CharChecker.isInvalid(ch) || CharChecker.isControl(ch)) {
                    continue;
                }
                if (CharChecker.isWhitespace(ch)) {
                    stringBuilder.append(" ");
                } else {
                    stringBuilder.append(ch);
                }
            }
            return stringBuilder.toString();
        }

        /* Runs basic whitespace cleaning and splitting on a piece of text. */
        static List<String> whitespaceTokenize(String text) {
            if (text == null) {
                throw new NullPointerException("The input String is null.");
            }
            return Arrays.asList(text.split(" "));
        }

        /* Splits punctuation on a piece of text. */
        static List<String> runSplitOnPunc(String text) {
            if (text == null) {
                throw new NullPointerException("The input String is null.");
            }

            List<String> tokens = new ArrayList<>();
            boolean startNewWord = true;
            for (int i = 0; i < text.length(); i++) {
                char ch = text.charAt(i);
                if (CharChecker.isPunctuation(ch)) {
                    tokens.add(String.valueOf(ch));
                    startNewWord = true;
                } else {
                    if (startNewWord) {
                        tokens.add("");
                        startNewWord = false;
                    }
                    tokens.set(tokens.size() - 1, Iterables.getLast(tokens) + ch);
                }
            }

            return tokens;
        }
    }
    static final class CharChecker {

        /** To judge whether it's an empty or unknown character. */
        public static boolean isInvalid(char ch) {
            return (ch == 0 || ch == 0xfffd);
        }

        /** To judge whether it's a control character(exclude whitespace). */
        public static boolean isControl(char ch) {
            if (Character.isWhitespace(ch)) {
                return false;
            }
            int type = Character.getType(ch);
            return (type == Character.CONTROL || type == Character.FORMAT);
        }

        /** To judge whether it can be regarded as a whitespace. */
        public static boolean isWhitespace(char ch) {
            if (Character.isWhitespace(ch)) {
                return true;
            }
            int type = Character.getType(ch);
            return (type == Character.SPACE_SEPARATOR
                    || type == Character.LINE_SEPARATOR
                    || type == Character.PARAGRAPH_SEPARATOR);
        }

        /** To judge whether it's a punctuation. */
        public static boolean isPunctuation(char ch) {
            int type = Character.getType(ch);
            return (type == Character.CONNECTOR_PUNCTUATION
                    || type == Character.DASH_PUNCTUATION
                    || type == Character.START_PUNCTUATION
                    || type == Character.END_PUNCTUATION
                    || type == Character.INITIAL_QUOTE_PUNCTUATION
                    || type == Character.FINAL_QUOTE_PUNCTUATION
                    || type == Character.OTHER_PUNCTUATION);
        }

        private CharChecker() {}
    }
    public final class FullTokenizer {
        private final BasicTokenizer basicTokenizer;
        private final WordpieceTokenizer wordpieceTokenizer;
        private final Map<String, Integer> dic;

        public FullTokenizer(Map<String, Integer> inputDic, boolean doLowerCase) {
            dic = inputDic;
            basicTokenizer = new BasicTokenizer(doLowerCase);
            wordpieceTokenizer = new WordpieceTokenizer(inputDic);
        }

        public List<String> tokenize(String text) {
            List<String> splitTokens = new ArrayList<>();
            for (String token : basicTokenizer.tokenize(text)) {
                splitTokens.addAll(wordpieceTokenizer.tokenize(token));
            }
            return splitTokens;
        }

        public List<Integer> convertTokensToIds(List<String> tokens) {
            List<Integer> outputIds = new ArrayList<>();
            for (String token : tokens) {
                outputIds.add(dic.get(token));
            }
            return outputIds;
        }
    }
    public final class WordpieceTokenizer {
        private final Map<String, Integer> dic;

        private static final String UNKNOWN_TOKEN = "[UNK]"; // For unknown words.
        private static final int MAX_INPUTCHARS_PER_WORD = 200;

        public WordpieceTokenizer(Map<String, Integer> vocab) {
            dic = vocab;
        }

        /**
         * Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first
         * algorithm to perform tokenization using the given vocabulary. For example: input = "unaffable",
         * output = ["un", "##aff", "##able"].
         *
         * @param text: A single token or whitespace separated tokens. This should have already been
         *     passed through `BasicTokenizer.
         * @return A list of wordpiece tokens.
         */
        public List<String> tokenize(String text) {
            if (text == null) {
                throw new NullPointerException("The input String is null.");
            }

            List<String> outputTokens = new ArrayList<>();
            for (String token : BasicTokenizer.whitespaceTokenize(text)) {

                if (token.length() > MAX_INPUTCHARS_PER_WORD) {
                    outputTokens.add(UNKNOWN_TOKEN);
                    continue;
                }

                boolean isBad = false; // Mark if a word cannot be tokenized into known subwords.
                int start = 0;
                List<String> subTokens = new ArrayList<>();

                while (start < token.length()) {
                    String curSubStr = "";

                    int end = token.length(); // Longer substring matches first.
                    while (start < end) {
                        String subStr =
                                (start == 0) ? token.substring(start, end) : "##" + token.substring(start, end);
                        if (dic.containsKey(subStr)) {
                            curSubStr = subStr;
                            break;
                        }
                        end--;
                    }

                    // The word doesn't contain any known subwords.
                    if ("".equals(curSubStr)) {
                        isBad = true;
                        break;
                    }

                    // curSubStr is the longeset subword that can be found.
                    subTokens.add(curSubStr);

                    // Proceed to tokenize the resident string.
                    start = end;
                }

                if (isBad) {
                    outputTokens.add(UNKNOWN_TOKEN);
                } else {
                    outputTokens.addAll(subTokens);
                }
            }

            return outputTokens;
        }
    }

    static public class Question {

        List<Item> data;
        String version;

        public Question() {
            data = new ArrayList<>();
        }

        public List<Item> getData() {
            return data;
        }

        public void setData(List<Item> data) {
            this.data = data;
        }

        public String getVersion() {
            return version;
        }

        public void setVersion(String version) {
            this.version = version;
        }

        public static class Item{
            String title;
            List<Paragraphs> paragraphs;

            public String getTitle() {
                return title;
            }

            public void setTitle(String title) {
                this.title = title;
            }

            public List<Paragraphs> getParagraphs() {
                return paragraphs;
            }

            public void setParagraphs(List<Paragraphs> paragraphs) {
                this.paragraphs = paragraphs;
            }
        }

        public static class Paragraphs{
            String context;
            List<Qas> qas;

            public String getContext() {
                return context;
            }

            public void setContext(String context) {
                this.context = context;
            }

            public List<Qas> getQas() {
                return qas;
            }

            public void setQas(List<Qas> qas) {
                this.qas = qas;
            }
        }

        public static class Qas{
            String id;
            String question;
            List<Answers> answers;

            public String getId() {
                return id;
            }

            public void setId(String id) {
                this.id = id;
            }

            public String getQuestion() {
                return question;
            }

            public void setQuestion(String question) {
                this.question = question;
            }

            public List<Answers> getAnswers() {
                return answers;
            }

            public void setAnswers(List<Answers> answers) {
                this.answers = answers;
            }
        }

        public static class Answers{
            int answer_start;
            String text;

            public int getAnswer_start() {
                return answer_start;
            }

            public void setAnswer_start(int answer_start) {
                this.answer_start = answer_start;
            }

            public String getText() {
                return text;
            }

            public void setText(String text) {
                this.text = text;
            }
        }
    }

    public static class Result {
        private List<Data> data;

        public void setData(List<Data> data) {
            this.data = data;
        }

        public List<Data> getData() {
            return data;
        }

        public static class Data{
            String result;
            String[] answers;

            public String getResult() {
                return result;
            }

            public void setResult(String result) {
                this.result = result;
            }

            public String[] getAnswers() {
                return answers;
            }

            public void setAnswers(String[] answers) {
                this.answers = answers;
            }
        }
    }
}
