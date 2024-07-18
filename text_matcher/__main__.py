import click
from text_matcher.matcher import Text, Matcher
import os
import glob
import csv
import logging
import itertools

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def getFiles(path):
    """ 
    Determines whether a path is a file or directory. 
    If it's a directory, it gets a list of all the text files 
    in that directory, recursively. If not, it gets the file. 
    """

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # Get list of all files in dir, recursively. 
        return glob.glob(path + "/**/*.txt", recursive=True)
    else:
        raise click.ClickException(f"The path {path} doesn't appear to be a file or directory")


def checkLog(logfile, textpair):
    """ 
    Checks the log file to make sure we haven't already done a particular analysis. 
    Returns True if the pair is in the log already. 
    """
    logging.debug('Looking in the log for textpair:' % textpair)
    if not os.path.isfile(logfile):
        logging.debug('No log file found.')
        return None

    with open(logfile, newline='') as f:
        pairs = [[row[0], row[1]] for row in csv.reader(f)]
    # logging.debug('Pairs already in log: %s' % pairs)
    return textpair in pairs


def createLog(logfile, columnLabels):
    """ 
    Creates a log file and sets up headers so that it can be easily read 
    as a CSV later. 
    """
    header = ','.join(columnLabels) + '\n'
    with open(logfile, 'w') as f:
        f.write(header)


@click.command()
@click.argument('text1')
@click.argument('text2')
@click.option('-t', '--threshold', type=int, default=3, \
              help='The shortest length of match to include in the list of initial matches.')
@click.option('-c', '--cutoff', type=int, default=5, \
              help='The shortest length of match to include in the final list of extended matches.')
@click.option('-n', '--ngrams', type=int, default=3, \
              help='The ngram n-value to match against.')
@click.option('-m', '--mindistance', type=int, default=8, \
              help='The minimum value for distance between two match.')
@click.option('-l', '--logfile', default='log.txt', help='The name of the log file to write to.')
@click.option('--stops', is_flag=True, help='Include stopwords in matching.', default=False)
@click.option('--verbose', is_flag=True, help='Enable verbose mode, giving more information.', default=False)
@click.option('--silent', is_flag=True, help='Disable printing of matches', default=False)
def cli(text1, text2, threshold, cutoff, ngrams, logfile, verbose, stops, mindistance, silent):
    """ This program finds similar text in two text files. """

    # Determine whether the given path is a file or directory.

    texts1 = getFiles(text1)
    texts2 = getFiles(text2)

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if stops:
        logging.debug('Including stopwords in tokenizing.')

    logging.debug(f'Comparing this/these text(s): {texts1}')
    logging.debug(f'with this/these text(s): {texts2}')

    pairs = list(itertools.product(texts1, texts2))

    numPairs = len(pairs)

    logging.debug(f'Comparing {numPairs} pairs.')
    # logging.debug('List of pairs to compare: %s' % pairs)

    logging.debug('Loading files into memory.')

    texts = {}
    for filename in texts1 + texts2:
        with open(filename, errors="ignore") as f:
            text = f.read()
        if filename not in texts:
            texts[filename] = text

    logging.debug('Loading complete.')

    prevTextObjs = {}
    for index, pair in enumerate(pairs, 1):
        timeStart = os.times().elapsed
        logging.debug(f'Now comparing pair {index} of {numPairs}.')
        logging.debug(f'Comparing {pair[0]} with {pair[1]}.')

        # Make sure we haven't already done this pair. 
        if checkLog(logfile, [pair[0], pair[1]]):
            logging.debug('This pair is already in the log. Skipping.')
            continue
        else:
            # This means that there isn't a log file. Let's set one up.
            logging.debug('No log file found. Setting one up.')
            # Set up columns and their labels. 
            columnLabels = ['Text A', 'Text B', 'Threshold', 'Cutoff', 'N-Grams', 'Num Matches', 'Text A Length',
                            'Text B Length', 'Locations in A', 'Locations in B']
            createLog(logfile, columnLabels)

        logging.debug('Processing texts.')

        # Put this in a dictionary so we don't have to process a file twice.
        for filename in pair:
            if filename not in prevTextObjs:
                logging.debug(f'Processing text: {filename}')
                prevTextObjs[filename] = Text(texts[filename], filename)

        # Just more convenient naming. 
        filenameA, filenameB = pair[0], pair[1]
        textObjA = prevTextObjs[filenameA]
        textObjB = prevTextObjs[filenameB]

        # Reset the table of previous text objects, so we don't overload memory. 
        # This means we'll only remember the previous two texts. 
        prevTextObjs = {filenameA: textObjA, filenameB: textObjB}

        # Do the matching. 
        myMatch = Matcher(textObjA, textObjB, threshold=threshold, cutoff=cutoff, ngramSize=ngrams,
                          removeStopwords=stops, minDistance=mindistance, silent=silent)
        myMatch.match()

        timeEnd = os.times().elapsed
        timeElapsed = timeEnd - timeStart
        logging.debug(f'Matching completed in {timeElapsed} seconds.')

        # Write to the log, but only if a match is found.
        if myMatch.numMatches > 0:
            logItems = [pair[0], pair[1], threshold, cutoff, ngrams, myMatch.numMatches, myMatch.textA.length,
                        myMatch.textB.length, str(myMatch.locationsA), str(myMatch.locationsB)]
            logging.debug(f'Logging items: {logItems}')
            line = ','.join(f'"{item}"' for item in logItems) + '\n'
            with open(logfile, 'a') as f:
                f.write(line)


if __name__ == '__main__':
    cli()
