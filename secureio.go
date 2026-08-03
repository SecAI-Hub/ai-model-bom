package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	maxConfigBytes   int64 = 1 << 20
	maxMetadataBytes int64 = 4 << 20
	maxEvidenceBytes int64 = 16 << 20
	maxBOMBytes      int64 = 64 << 20
	maxSecretBytes   int64 = 64 << 10
)

func readRegularFile(path string, maxBytes int64) ([]byte, error) {
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("path is required")
	}
	if maxBytes <= 0 {
		return nil, errors.New("maximum size must be positive")
	}

	pathInfo, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !pathInfo.Mode().IsRegular() {
		return nil, fmt.Errorf("refusing non-regular file %q", path)
	}
	if pathInfo.Size() > maxBytes {
		return nil, fmt.Errorf("file %q exceeds %d-byte limit", path, maxBytes)
	}

	// #nosec G304 -- this is the centralized bounded reader for an explicit
	// operator-selected input; Lstat/Fstat/path-after identity checks surround it.
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	before, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || !os.SameFile(pathInfo, before) ||
		pathInfo.Size() != before.Size() || !pathInfo.ModTime().Equal(before.ModTime()) {
		return nil, fmt.Errorf("file %q changed while being opened", path)
	}

	data, err := io.ReadAll(io.LimitReader(f, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("file %q exceeds %d-byte limit", path, maxBytes)
	}
	after, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return nil, fmt.Errorf("file %q changed while being read", path)
	}
	pathAfter, err := os.Lstat(path)
	if err != nil || !os.SameFile(after, pathAfter) || after.Size() != pathAfter.Size() ||
		!after.ModTime().Equal(pathAfter.ModTime()) {
		return nil, fmt.Errorf("file path %q changed while being read", path)
	}
	return data, nil
}

func decodeStrictJSON(data []byte, dst interface{}) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if err := ensureJSONEOF(dec); err != nil {
		return err
	}
	return nil
}

func ensureJSONEOF(dec *json.Decoder) error {
	var extra interface{}
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return nil
}

func decodeStrictYAML(data []byte, dst interface{}) error {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var extra interface{}
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple YAML documents are not allowed")
		}
		return fmt.Errorf("trailing YAML: %w", err)
	}
	return nil
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("output path is required")
	}
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() {
			return fmt.Errorf("refusing to replace non-regular output %q", path)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".ai-model-bom-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}
	defer cleanup()

	if err := tmp.Chmod(mode); err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}

	// #nosec G304 -- dir is the parent of an explicit CLI output and is opened
	// only to fsync the already completed atomic rename.
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

func writeNewFile(path string, data []byte, mode os.FileMode) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("output path is required")
	}
	// #nosec G304 -- path is an explicit CLI key-output path; O_CREATE|O_EXCL
	// prevents following or replacing an existing file or symlink.
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		_ = f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	if _, err := f.Write(data); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	if err := dir.Sync(); err != nil {
		_ = dir.Close()
		return err
	}
	if err := dir.Close(); err != nil {
		return err
	}
	ok = true
	return nil
}
