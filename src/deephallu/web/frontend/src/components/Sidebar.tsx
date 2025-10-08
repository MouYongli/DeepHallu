"use client"

import { useState } from "react"
import {
  Folder,
  FolderOpen,
  Image,
  Upload,
  ChevronRight,
  ChevronDown,
  X
} from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@radix-ui/react-collapsible"
import { Separator } from "@radix-ui/react-separator"
import { cn } from "@/lib/utils"
import { mockImages } from "@/lib/mockData"
import { ImageMetadata } from "@/lib/types"

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onSelectImage?: (imageId: string) => void
  selectedImageId?: string
}

interface FileItem {
  name: string
  type: 'file' | 'folder'
  path: string
  children?: FileItem[]
  id?: string
  metadata?: ImageMetadata
}

// Generate file structure from mock data
const generateFileStructure = (): FileItem[] => {
  const mmeCategories: { [key: string]: FileItem[] } = {}

  mockImages.forEach(image => {
    if (image.dataset === 'MME') {
      if (!mmeCategories[image.category]) {
        mmeCategories[image.category] = []
      }
      mmeCategories[image.category].push({
        name: image.name,
        type: 'file',
        path: image.path,
        id: image.id,
        metadata: image
      })
    }
  })

  const mmeChildren = Object.entries(mmeCategories).map(([category, files]) => ({
    name: category,
    type: 'folder' as const,
    path: `/dataset/mme_images/${category.toLowerCase().replace(' ', '_')}`,
    children: files
  }))

  return [
    {
      name: "MME Dataset",
      type: "folder",
      path: "/dataset/mme",
      children: mmeChildren
    }
  ]
}

function FileTreeItem({
  item,
  level = 0,
  onSelectImage,
  selectedImageId
}: {
  item: FileItem
  level?: number
  onSelectImage?: (imageId: string) => void
  selectedImageId?: string
}) {
  const [isOpen, setIsOpen] = useState(level === 0)
  const [showTooltip, setShowTooltip] = useState(false)
  const hasChildren = item.children && item.children.length > 0
  const isSelected = item.id === selectedImageId

  const handleClick = () => {
    if (item.type === 'file' && item.id && onSelectImage) {
      onSelectImage(item.id)
    }
  }

  return (
    <div>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <div
            className={cn(
              "relative flex items-center space-x-2 px-2 py-1.5 rounded cursor-pointer text-sm transition-colors",
              isSelected
                ? "bg-blue-100 dark:bg-blue-900/50 text-blue-900 dark:text-blue-100"
                : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
            )}
            style={{ paddingLeft: `${level * 16 + 8}px` }}
            onClick={item.type === 'file' ? handleClick : undefined}
            onMouseEnter={() => item.type === 'file' && setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
          >
            {hasChildren && (
              <>
                {isOpen ? (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-gray-500" />
                )}
              </>
            )}
            {!hasChildren && <div className="w-4" />}

            {item.type === 'folder' ? (
              isOpen ? (
                <FolderOpen className="h-4 w-4 text-blue-500" />
              ) : (
                <Folder className="h-4 w-4 text-blue-500" />
              )
            ) : (
              <Image className="h-4 w-4 text-green-500" />
            )}

            <span className="truncate flex-1">
              {item.name}
            </span>

            {/* Tooltip on hover */}
            {showTooltip && item.metadata && (
              <div className="absolute left-full ml-2 top-0 z-50 bg-gray-900 dark:bg-gray-800 text-white text-xs rounded-lg shadow-lg p-3 min-w-[200px] border border-gray-700">
                <div className="space-y-1">
                  <div className="font-semibold border-b border-gray-700 pb-1 mb-2">
                    {item.metadata.name}
                  </div>
                  <div><span className="text-gray-400">Dataset:</span> {item.metadata.dataset}</div>
                  <div><span className="text-gray-400">Category:</span> {item.metadata.category}</div>
                  <div><span className="text-gray-400">Format:</span> {item.metadata.format}</div>
                  <div><span className="text-gray-400">Resolution:</span> {item.metadata.resolution}</div>
                  <div><span className="text-gray-400">Size:</span> {item.metadata.size}</div>
                </div>
              </div>
            )}
          </div>
        </CollapsibleTrigger>

        {hasChildren && (
          <CollapsibleContent>
            {item.children?.map((child, index) => (
              <FileTreeItem
                key={index}
                item={child}
                level={level + 1}
                onSelectImage={onSelectImage}
                selectedImageId={selectedImageId}
              />
            ))}
          </CollapsibleContent>
        )}
      </Collapsible>
    </div>
  )
}

export function Sidebar({ isOpen, onClose, onSelectImage, selectedImageId }: SidebarProps) {
  const [dragOver, setDragOver] = useState(false)
  const [uploadedImages, setUploadedImages] = useState<FileItem[]>([])
  const [uploading, setUploading] = useState(false)
  const fileStructure = generateFileStructure()

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    handleFiles(files)
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    handleFiles(files)
    e.target.value = ''
  }

  const handleFiles = async (files: File[]) => {
    const imageFiles = files.filter(file => file.type.startsWith('image/'))

    if (imageFiles.length === 0) {
      alert('Please upload only image files (JPG, PNG, GIF, etc.)')
      return
    }

    setUploading(true)

    try {
      const newImages: FileItem[] = []

      for (const file of imageFiles) {
        if (file.size > 10 * 1024 * 1024) {
          alert(`File ${file.name} is too large. Maximum size is 10MB.`)
          continue
        }

        const objectUrl = URL.createObjectURL(file)
        const id = `uploaded_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

        // Load image to get dimensions
        const resolution = await new Promise<string>((resolve) => {
          const img = new window.Image()
          img.onload = () => {
            resolve(`${img.width}x${img.height}`)
          }
          img.onerror = () => {
            resolve('Unknown')
          }
          img.src = objectUrl
        })

        const metadata: ImageMetadata = {
          id,
          name: file.name,
          path: objectUrl,
          category: 'Uploaded',
          dataset: 'User Upload',
          resolution,
          format: file.name.split('.').pop()?.toUpperCase() || 'Unknown',
          size: formatFileSize(file.size),
          questions: [
            { question: "Describe what you see in this image.", answer: "" }
          ]
        }

        const newImage: FileItem = {
          id,
          name: file.name,
          type: 'file',
          path: objectUrl,
          metadata
        }

        newImages.push(newImage)
        mockImages.push(metadata)
      }

      setUploadedImages(prev => [...prev, ...newImages])

      if (newImages.length === 1 && onSelectImage) {
        onSelectImage(newImages[0].id!)
      }

    } catch (error) {
      console.error('Upload failed:', error)
      alert('Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  if (!isOpen) return null

  return (
    <div className="w-80 h-full bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      {/* Sidebar Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Images
        </h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
        >
          <X className="h-4 w-4 text-gray-500" />
        </button>
      </div>

      {/* Upload Area */}
      <div className="p-4">
        <div
          className={cn(
            "border-2 border-dashed rounded-lg p-4 text-center transition-colors",
            dragOver
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
              : "border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500",
            uploading && "opacity-50 pointer-events-none"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {uploading ? (
            <>
              <div className="animate-spin h-6 w-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-xs text-blue-600 dark:text-blue-400">
                Uploading...
              </p>
            </>
          ) : (
            <>
              <Upload className="h-6 w-6 text-gray-400 mx-auto mb-2" />
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                Drop images or
              </p>
              <label htmlFor="file-upload" className="cursor-pointer">
                <span className="text-blue-600 hover:text-blue-700 text-xs font-medium">
                  browse
                </span>
                <input
                  id="file-upload"
                  type="file"
                  className="hidden"
                  multiple
                  accept="image/*"
                  onChange={handleFileUpload}
                />
              </label>
            </>
          )}
        </div>
      </div>

      <Separator className="bg-gray-200 dark:bg-gray-700 h-px" />

      {/* File Tree */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          {/* Uploaded Images Section */}
          {uploadedImages.length > 0 && (
            <div className="mb-2">
              <FileTreeItem
                item={{
                  name: `Uploaded (${uploadedImages.length})`,
                  type: 'folder',
                  path: '/uploaded',
                  children: uploadedImages
                }}
                onSelectImage={onSelectImage}
                selectedImageId={selectedImageId}
              />
            </div>
          )}

          {/* Dataset Structure */}
          {fileStructure.map((item, index) => (
            <FileTreeItem
              key={index}
              item={item}
              onSelectImage={onSelectImage}
              selectedImageId={selectedImageId}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
