import com.juulcrienen.githubapiwrapper.GitHubAPIWrapper;
import com.juulcrienen.githubapiwrapper.helpers.FileHelper;
import helpers.DocumentationParser;
import org.eclipse.jgit.api.CloneCommand;
import org.eclipse.jgit.api.Git;
import org.kohsuke.github.GHContent;
import org.kohsuke.github.GHRepository;
import org.kohsuke.github.GHUser;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

import static helpers.DocumentationParser.getRepositories;

public class GetDocumentation {
    static DocumentationParser parser = new DocumentationParser();

    public static void main(String[] args) throws IOException {
        GitHubAPIWrapper wrapper = new GitHubAPIWrapper();

        List<String> repositoriesName = getRepositories();
        List<GHRepository> repos = wrapper.getGitHubRepositories(repositoriesName);

        for (GHRepository repo : repos) {
            String destination = "documentation/" + repo.getName() + "-" + repo.getOwner().getLogin();

//            Files.createDirectories(Paths.get(destination));
            File newDirectory = new File(destination);
            newDirectory.mkdir();
            writeReadMeToFile(repo.getReadme(), repo.getName(), destination);
            if(repo.hasWiki()) {
                getWiki(repo.getOwner(), repo.getName());
                mergeWiki("wiki_" + repo.getName(), destination);
            }
//          Mind commenting this if you don't need it, since it takes a while to compute
            getComments(repo, destination);
        }

    }

    /**
     * Copy the readme file to a txt file
     * @param readme - the readme file in question
     * @param repoName - name of the repository/project
     */
    private static void writeReadMeToFile(GHContent readme, String repoName, String filePath) {
        try {
            String readmeString = null;
            try (Scanner scanner = new Scanner(readme.read(), StandardCharsets.UTF_8.name())) {
                readmeString = scanner.useDelimiter("\\A").next();
            }

            readmeString = parser.filterReadmeFile(readmeString);

//            String filePath = "documentation/readme/readme_" + repoName;

            byte[] byteContent = readmeString.getBytes();

            Path path = Paths.get(filePath + "/readme.txt");

            Files.write(path, byteContent);
            System.out.println("Readme file saved successfully at " + filePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * TODO: Implement a comment parser for languages that use different comment style than Java, Haskell, Python
     * Idea: depending on the language, use a regex to find comments and then save them to a file. It should combine the comments from all languages in a single file.
     * @param repo - the repository for which the comments are to be found and then saved in an external file
     *
     */
    private static void getComments(GHRepository repo, String destination) {
        try {
            Set<String> languages = repo.listLanguages().keySet().stream().filter(language -> parser.languageToExtension.containsKey(language.toLowerCase())).collect(Collectors.toSet());

            StringBuilder completeCommentFile = new StringBuilder();
            for(String language : languages) {
                String value = parser.languageToExtension.get(language.toLowerCase());
                List<File> contents = FileHelper.getFiles(repo, value, null);
                StringBuilder childComments = new StringBuilder();

                //                TODO: C# documentation comment, R, Ruby, PHP
                //                TODO: Make javadoc style of comment more useful for comparing comments

                if (List.of("java", "c++", "c#", "scala", "kotlin", "javascript", "typescript", "rust", "swift", "go").contains(language.toLowerCase())) {
                    for (File file : contents) {
                        childComments.append(parser.readJavaStyleComment(file));
                    }
                } else if (language.equalsIgnoreCase("python")) {
                    for (File file : contents) {
                        childComments.append(parser.readPythonStyleComment(file));
                    }
                } else if (language.equalsIgnoreCase("haskell")) {
                    for (File file : contents) {
                        childComments.append(parser.readHaskellComment(file));
                    }
                }

                completeCommentFile.append(childComments);
            }
            writeCommentsToFile(completeCommentFile.toString(), destination);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Write the content of the identified comments to an external file
     * @param content - the content of the comments
     */
    public static void writeCommentsToFile(String content, String path) {
//        String path = "documentation/comments/comments_" + repoName;
        try (FileWriter writer = new FileWriter(path + "/comments.txt")) {
            writer.write(content);
            System.out.println("Combined comment file saved successfully at: " + path);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Wiki its by itself a different repository, and there are no APIs to access it. The procedure is the following: clone wiki repo, then combine the .md files, and if possible delete the repository
     * @param owner - owner of the repo, a GHUser
     * @param repoName - name of repository/project of which the wiki is wanted
     */
    private static void getWiki(GHUser owner, String repoName) {
        String ownerName = owner.getLogin();
        String repoURL = "https://github.com/" + ownerName + "/" + repoName + ".wiki.git";
        String filePath = "wikiTemp/wiki_" + repoName;
        try {
            CloneCommand cloneCommand = Git.cloneRepository()
                    .setURI(repoURL)
                    .setDirectory(new File(filePath));

            cloneCommand.call().close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * Combine the obtained .md files from the Wiki repo
     * @param folderName - the name of the folder where the repo was saved
     */
    private static void mergeWiki(String folderName, String new_destination) {
        String filePath = "wikiTemp/" + folderName;
        File folder = new File(filePath);
        if (folder.isDirectory()) {
            File[] files = folder.listFiles();
            if(files != null) {
                if (files.length > 2 ){
                    //ignore .git folder
                    mergeFiles(Arrays.copyOfRange(files, 1, files.length), new_destination);
                }
                deleteFolder(folder);
            }
        }
    }

    /**
     * Merge the .md files into a single one, that has the name of the repo
     * @param files - the files to be merged
     * @param file_name - name of the resulting file
     */
    private static void mergeFiles(File[] files, String file_name) {
        StringBuilder finalFile = new StringBuilder();
        for (File file : files) {
            if (file.isFile())
                finalFile.append(readFile(file));
            finalFile.append(System.lineSeparator());
        }
//        System.out.println(file_name);

        writeWikiToFile(finalFile.toString(), file_name);
    }

    /**
     * Read the components of each file
     */
    private static String readFile(File file) {
        StringBuilder content = new StringBuilder();
        try {
            Path filePath = file.toPath();
            Files.lines(filePath, StandardCharsets.UTF_8).forEach(line -> {
                content.append(line).append(System.lineSeparator());
            });
        } catch (IOException e) {
            e.printStackTrace();
        }

        return parser.filterReadmeFile(content.toString());
    }


    /**
     * Write the combined Wiki files into a single file
     * @param content - the content of all wiki pages combined into a single one
     * @param destination - location where to write the content
     */
    private static void writeWikiToFile(String content, String destination) {
        try (FileWriter writer = new FileWriter(destination + "/wiki.txt")) {
            writer.write(content);
            System.out.println("Combined wiki files saved successfully at: " + destination);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Method used to delete the repo of a wiki after it got merged into a single file
     * @param folder - folder location
     */
    private static void deleteFolder(File folder) {
        if (folder.exists()) {
            File[] files = folder.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory())
                        deleteFolder(file);
                    else
                        file.delete();
                }
            }
            folder.delete();
        }
    }




}
