package helpers;

import com.vladsch.flexmark.ast.*;
import com.vladsch.flexmark.parser.Parser;
import com.vladsch.flexmark.util.ast.*;
import org.jetbrains.annotations.NotNull;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;
import org.jsoup.select.NodeTraversor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DocumentationParser {

    public final Map<String, String> languageToExtension = new HashMap<>();
    public final List<String> wordsToFilter = new ArrayList<>();
    private final String singleLineCommentRegexJava = "//.*|/\\*.*\\*/";
    private final String multiLineCommentStartRegex = "^\\s*/\\*.*$";
    private final String multiLineCommentEndRegex = ".*\\*/\\s*$";
    private final String commentRegexPython = "#.*|'''(?:.|\\\\n)*?'''";
    private final String commentRegexHaskell = "--.*|\\{-(?:.|\\n)*?-\\}";

    public DocumentationParser() {
        populateMap();
        populateList();
    }

    /**
     * Read files and look for comments. Supports file extensions that have similar comment symbols as Java
     * @param file - the file that needs to be read
     * @return a string containing the text components of the comments in the file
     */
    public String readJavaStyleComment(File file) {
        StringBuilder extracted = new StringBuilder();
        boolean inMultiLineComment = false;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line = reader.readLine();
            while (line != null) {
                Pattern singleLinePattern = Pattern.compile(singleLineCommentRegexJava);
                Matcher singleLineMatcher = singleLinePattern.matcher(line);
                if (singleLineMatcher.find() && !inMultiLineComment)
                    extracted.append(singleLineMatcher.group()).append(System.lineSeparator());
                else {
                    Pattern multiLineStartPattern = Pattern.compile(multiLineCommentStartRegex);
                    Matcher multiLineStartMatcher = multiLineStartPattern.matcher(line);
                    if (multiLineStartMatcher.matches()) {
                        inMultiLineComment = true;
                        extracted.append(line).append(System.lineSeparator());
                    }

                    Pattern multiLineEndPatten = Pattern.compile(multiLineCommentEndRegex);
                    Matcher multiLineEndMatcher = multiLineEndPatten.matcher(line);
                    if (multiLineEndMatcher.matches()) {
                        inMultiLineComment = false;
                        extracted.append(line).append(System.lineSeparator());
                    }

                    if (inMultiLineComment) {
                        extracted.append(line).append(System.lineSeparator());
                    }
                }
                line = reader.readLine();
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return removeCommonWords(removeCommentSymbols(extracted.toString()));
    }

    /**
     * Read files and look for comments. Supports file extensions that have similar comment symbols as Python
     * @param file - the file that needs to be read
     * @return a string containing the text components of the comments in the file
     */
    public String readPythonStyleComment(File file) {
        StringBuilder extracted = new StringBuilder();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line = reader.readLine();

            while (line != null) {
                Pattern pattern = Pattern.compile(commentRegexPython, Pattern.MULTILINE);
                Matcher matcher = pattern.matcher(line);

                if(matcher.find()) {
                    extracted.append(matcher.group()).append(System.lineSeparator());
                }
                line = reader.readLine();
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return removeCommonWords(removeCommentSymbols(extracted.toString()));
    }

    /**
     * Read files and look for comments in Haskell source code.
     * Should constructs of the form {-# smth #-} be considered comments or not?
     * @param file - the file that needs to be read
     * @return a string containing the text components of the comments in the file
     */
    public String readHaskellComment(File file) {
        StringBuilder extracted = new StringBuilder();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line = reader.readLine();

            while (line != null) {
                Pattern pattern = Pattern.compile(commentRegexHaskell, Pattern.MULTILINE);
                Matcher matcher = pattern.matcher(line);

                if (matcher.find()) {
                    extracted.append(matcher.group()).append(System.lineSeparator());
                }
                line = reader.readLine();

            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
//      Moved text processing to python
        return extracted.toString();
//        removeCommentSymbols(extracted.toString()));
    }

    /**
     * Eliminate comment symbols and additional empty spaces
     * TODO: support for comment symbols from Python, Haskell, R etc
     * @param text - text content of a file
     * @return filtered string containing the file's text
     */
    public String removeCommentSymbols(String text) {
        text = text.replaceAll("[/*]", "");
//        text = text.replaceAll(" {2,}", " ");
        text = text.replaceAll("#", "");
        text = text.replaceAll("--|\\{-|\\-\\}", "");
        text = text.trim();
        return text;
    }

    /**
     * Remove very common words from file (also known as stop words) and then remove the created empty space
     * @param text - text content of a file
     * @return filtered string containing the file's text
     */
    public String removeCommonWords(String text) {
        for (String word : wordsToFilter) {
            text = text.replaceAll("\\b" + Pattern.quote(word) + "\\b", "");
        }
//        text = text.replaceAll(" {2,}", " ");
        text = text.trim();
        return text;

    }

    /**
     * Remove unwanted/ not useful piece of text from a Readme file (or .md in general)
     * @param readme - text content of a file
     * @return filtered string containing the file's text
     */
    public String filterReadmeFileOld(String readme) {
//      Remove links
        readme = readme.replaceAll("!\\[[^\\]]*]\\([^)]*\\)", "");
        readme = readme.replaceAll("\\[[^\\]]*]\\([^)]*\\)", "");

//      Remove code blocks, keep inline code
        readme = readme.replaceAll("```.*?```", "");
        readme = readme.replaceAll("~~~.*?~~~", "");
//        readme = readme.replaceAll("`.*?`", "");

//      Remove images
        readme = readme.replaceAll("!\\[[^\\]]*]\\([^)]*\\)", "");

//      Remove headers and lists
        readme = readme.replaceAll("#", "");
        readme = readme.replaceAll("\\*", "");

//      Remove excess new lines
        readme = readme.replaceAll("\\n{2,}", "\n");
        return removeCommonWords(readme);
    }

    /**
     * Updated function to remove unwanted/ not useful piece of text from a Readme file (or .md in general)
     * @param readme - text content of a file
     * @return filtered string containing the file's text
     */
    public String filterReadmeFile(String readme) {
        Node doc = Parser.builder().build().parse(readme);
        return extractText(doc) + extractHtml(readme);
    }

    /**
     * Extracts the html text from a md file
     * @param readme - text content of a file
     * @return string representation of the text between html tags
     */
    private static String extractHtml(String readme) {
        org.jsoup.nodes.Document jsoup = Jsoup.parse(readme);

        StringBuilder extracted = new StringBuilder();
        NodeTraversor.traverse(new org.jsoup.select.NodeVisitor() {
            @Override
            public void head(org.jsoup.nodes.@NotNull Node node, int i) {
                if (node instanceof Element) {
                    Element element = (Element) node;
                    extracted.append(element.text()).append(" ");
                }
            }
        }, jsoup);
        return extracted.toString().trim();
    }

    /**
     * selects only wanted parts of a md file
     * @param doc parsed md
     * @return string representation of the text
     */
    private static String extractText(Node doc) {
        StringBuilder extracted = new StringBuilder();
        new NodeVisitor(new VisitHandler<>(Heading.class, heading -> {
            extracted.append(heading.getText());
        })).visit(doc);

        new NodeVisitor(new VisitHandler<>(Paragraph.class, paragraph -> {
            for (Node child : paragraph.getChildren()) {
                if (!(child instanceof Image) && !(child instanceof Link)) {
                    extracted.append(child.getChars());

                }
            }
        })).visit(doc);

        new NodeVisitor(new VisitHandler<>(ListBlock.class, listBlock -> {
            extracted.append(extractText(listBlock)).append(System.lineSeparator());
        })).visit(doc);

        new NodeVisitor(new VisitHandler<>(Emphasis.class, emphasis -> {
            extracted.append(emphasis.getText()).append(System.lineSeparator());
        })).visit(doc);

        new NodeVisitor(new VisitHandler<>(StrongEmphasis.class, strongEmphasis -> {
            extracted.append(strongEmphasis.getText()).append(System.lineSeparator());
        })).visit(doc);

        new NodeVisitor(new VisitHandler<>(BlockQuote.class, blockQuote -> {
            extracted.append(blockQuote.getContentChars()).append(System.lineSeparator());
        })).visit(doc);

        new NodeVisitor(new VisitHandler<>(Link.class, link -> {
            extracted.append(link.getUrl()).append(System.lineSeparator());
        })).visit(doc);
        return extracted.toString();
    }

    /**
     * Read from file the list of repositories from which to extract data
     * @return - list of the repositories
     */
    public static List<String> getRepositories(boolean source) {
        List<String> repos = new ArrayList<>();
        String path = source ? "resources/repositories" : "resources/crosssim";
        try {
            File file = new File(path);
            Scanner reader = new Scanner(file);
            int label = -1;
            while(reader.hasNextLine()) {
                String line = reader.nextLine();
                if(!line.startsWith("#")) {
                    if (source)
                        repos.add(line + "#####" + label);
                    else
                        repos.add(line);
                }
                else
                    label ++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return repos;

    }

    // Tried to use Apache Lucene to get the list of end words, but I couldn't find the exact package
    private void populateList() {
        wordsToFilter.add("and");
        wordsToFilter.add("so");
        wordsToFilter.add("to");
        wordsToFilter.add("method");
        wordsToFilter.add("function");
        wordsToFilter.add("variable");
        wordsToFilter.add("the");
        wordsToFilter.add("or");
        wordsToFilter.add("be");
        wordsToFilter.add("but");
        wordsToFilter.add("in");
        wordsToFilter.add("that");
        wordsToFilter.add("of");
        wordsToFilter.add("a");
        wordsToFilter.add("an");
        wordsToFilter.add("have");
        wordsToFilter.add("too");
        wordsToFilter.add("has");
        wordsToFilter.add("it");
        wordsToFilter.add("I");
        wordsToFilter.add("that");
        wordsToFilter.add("you");
        wordsToFilter.add("she");
        wordsToFilter.add("he");
        wordsToFilter.add("they");
        wordsToFilter.add("them");
        wordsToFilter.add("with");
        wordsToFilter.add("on");
        wordsToFilter.add("do");
        wordsToFilter.add("this");
        wordsToFilter.add("that");
        wordsToFilter.add("can");
        wordsToFilter.add("by");
        wordsToFilter.add("what");
        wordsToFilter.add("who");
        wordsToFilter.add("where");
        wordsToFilter.add("why");
        wordsToFilter.add("such");
        wordsToFilter.add("are");
        wordsToFilter.add("as");
        wordsToFilter.add("no");
        wordsToFilter.add("not");

    }
    // A small dictionary population with popular programming languages and their extensions
    private void populateMap() {
        this.languageToExtension.put("java", "java");
        this.languageToExtension.put("python", "py");
        this.languageToExtension.put("c++", "cpp");
        this.languageToExtension.put("c", "c");
        this.languageToExtension.put("javascript", "js");
        this.languageToExtension.put("ruby", "rb");
        this.languageToExtension.put("c#", "cs");
        this.languageToExtension.put("swift", "swift");
        this.languageToExtension.put("go", "go");
        this.languageToExtension.put("typescript", "ts");
        this.languageToExtension.put("php", "php");
        this.languageToExtension.put("kotlin", "kt");
        this.languageToExtension.put("rust", "rs");
        this.languageToExtension.put("r", "r");
        this.languageToExtension.put("scala", "scala");
        this.languageToExtension.put("haskell", "hs");
    }

}
