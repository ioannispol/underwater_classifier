import os
import sys
import shutil
import datetime
from PIL import Image
from PIL.ExifTags import TAGS

# Check first if the given path exists before continuing to the next step.
while True:
    folderpath = raw_input('Folder path: ')
    print "\nChecking if folder path exists...\n"
    if os.path.exists(folderpath):
        print "Ok!\n"
        break;
    else:
        print "%s does not exist. Please enter a valid path.\n" % folderpath

filename_pattern = raw_input('Filename pattern: ')

def get_exif_data(filename):
    """Get embedded EXIF data from image file.

    Source: <a href="http://www.endlesslycurious.com/2011/05/11/extracting-image-">http://www.endlesslycurious.com/2011/05/11/extract...</a>             exif-data-with-python/
    """
    ret = {}
    try:
        img = Image.open(filename)
        if hasattr( img, '_getexif' ):
            exifinfo = img._getexif()
            if exifinfo != None:
                for tag, value in exifinfo.items():
                    decoded = TAGS.get(tag, tag)
                    ret[decoded] = value
    except IOError:
        print 'IOERROR ' + filename
    return ret

def get_date_taken(filename):
    datestring = get_exif_data(current_file)['DateTimeOriginal']
    return datetime.datetime.strptime(datestring, '%Y:%m:%d %H:%M:%S')

def get_filenames():
    os.chdir(folderpath)
    return os.listdir(os.getcwd())

def get_numbering_format(digits, num):
    if digits == 1:
        numberfmt = '00%s' % num
    elif digits == 2:
        numberfmt = '0%s' % num
    else:
        numberfmt = '%s' % num
    return numberfmt

def date_to_string(dateobj, format):
    return datetime.datetime.strftime(dateobj, format)

def multi_replace(text, dictobj):
    """Replace characters in the text based on the given dictionary."""
    for k, v in dictobj.iteritems():
        text = text.replace(k, v)
    return text

if __name__ == '__main__':
    filenames = get_filenames()
    for i in xrange(len(filenames)):
        num = i + 1
        digits = len(str(num))
        current_file = filenames[i]

        # Only rename files, ignore directories.
        if not os.path.isdir(current_file):
            # Key, value pairs of what to replace.
            dictobj = {
                '<num>': get_numbering_format(digits, num),
                '<datetaken>': date_to_string(get_date_taken(current_file),
                                              '%Y%m%d')
            }
            new_filename = multi_replace(filename_pattern, dictobj)
            shutil.move(current_file, new_filename)</datetaken></num>
