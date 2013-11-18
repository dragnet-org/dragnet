#!/bin/bash

set -e

if [ -z "$1" ]
then
    echo "Convert CETR to dragnet format"
    echo ""
    echo "Usage: $0 rootdir"
    exit 1
fi

ROOTDIR=$1

for D in en zh bbc freep myriad nypost nytimes reuters suntimes techweb tribune
do

    if [[ "$D" == "en" || "$D" == "zh" ]]
    then
        CLEANEVAL=1
        DATADIR=$1/cleaneval/$D/
    else
        CLEANEVAL=0
        DATADIR=$1/news/$D/
    fi

    echo "Processing $D"
    
    # for news, CETR has a slightly different directory structure
    if [[ $CLEANEVAL == 0 ]]
    then
        mkdir $DATADIR
        mkdir $DATADIR/gold
        mkdir $DATADIR/original
        mv $1/news/gold/$D/*.txt $DATADIR/gold
        mv $1/news/original/$D/*.html $DATADIR/original
    fi

    pushd $DATADIR
    
    # rename HTML, copy over all .html files, delete the rest
    mkdir HTML
    mv original/*.html HTML
    rm -r original
    
    # rename/move the corrected files
    mkdir Corrected
    ls gold/*.txt | perl -pe 'm/\/([0-9]+)\.txt/; system "cp gold/$1.txt Corrected/$1.html.corrected.txt\n";'
    rm -r gold
    
    # block_corrected
    mkdir block_corrected
    
    # remove all files in HTML that are not in Corrected
    # and vice versa
    perl -e '
    @html_files = glob "HTML/*";
    @corrected_files = glob "Corrected/*";
    %html = ();
    foreach (@html_files) {
        $m/\/([0-9]+)\.html/;
        $html{$1} = 1;
    }
    
    %corrected = ();
    foreach (@corrected_files) {
        $m/\/([0-9]+)\.html/;
        $corrected{$1} = 1;
    }
    
    # now check html.  if not in corrected then delete
    while ( ($key, $value) = each(%html) ) {
        if (!defined $corrected{$key}) {
            system "rm HTML/$key.html";
        }
    }
    
    # vice versa
    while ( ($key, $value) = each(%corrected) ) {
        if (!defined $html{$key}) {
            system "rm Corrected/$key.html.corrected.txt";
        }
    }
    '
    
    
    # now need to massage the formatting of the files
    if [[ $CLEANEVAL == 1 ]]
    then
        # need to remove the first line of Corrected file
        # this contains the URL, and we don't need it
        echo "Cleaning corrected files"
        ls Corrected/* | perl -pe '
        $orig_file = $_;
        open (F, $orig_file);
        open (FOUT, ">t");
        $k=0;
        while (<F>) {
            if ($k > 0) {
                print FOUT $_;
            }
            $k++;
        }
        close(FOUT);
        close(F);
        system "mv t $orig_file";
        '
    fi
    
    
    # clean up some things
    rm -r vips || true
    rm -r stripped || true
    
    popd

done

rm -r $1/news/gold
rm -r $1/news/original
rm -r $1/news/vips

